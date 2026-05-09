import argparse
import json
import statistics
from typing import Dict, List

import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import init_dist
from deep_ep.utils.gate import get_unbalanced_scores
from deep_ep.utils.math import per_token_cast_to_fp8


def percentile(values: List[float], q: float) -> float:
    values = sorted(values)
    return values[max(0, min(len(values) - 1, int(q * len(values)) - 1))]


def time_cuda(fn, warmups: int, iters: int) -> Dict[str, float]:
    torch.cuda.synchronize()
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1e3)
    return {
        "median_s": statistics.median(times),
        "p90_s": percentile(times, 0.9),
        "min_s": min(times),
        "max_s": max(times),
    }


def make_send_buffer(rank: int, num_ranks: int, bytes_per_peer: int, check: bool) -> torch.Tensor:
    send = torch.empty((num_ranks * bytes_per_peer,), dtype=torch.uint8, device="cuda")
    if check:
        send_2d = send.view(num_ranks, bytes_per_peer)
        for dst_rank in range(num_ranks):
            send_2d[dst_rank].fill_((rank * 17 + dst_rank) % 251)
    else:
        send.fill_(rank % 251)
    return send


def check_output(out: torch.Tensor, rank: int, bytes_per_peer: int) -> None:
    sample_idx = torch.tensor(
        [0, bytes_per_peer // 2, bytes_per_peer - 1],
        dtype=torch.long,
        device=out.device,
    )
    for src_rank in range(out.size(0)):
        expected = torch.full(
            (sample_idx.numel(),),
            (src_rank * 17 + rank) % 251,
            dtype=torch.uint8,
            device=out.device,
        )
        actual = out[src_rank, sample_idx]
        assert torch.equal(actual, expected), (rank, src_rank, actual, expected)


def make_dispatch_inputs(rank: int, args: argparse.Namespace):
    torch.manual_seed(args.seed + rank)
    scores = get_unbalanced_scores(
        args.num_tokens, args.num_experts, args.num_ranks or args.num_processes,
        args.num_topk, args.unbalanced_ratio, False)
    topk_weights, topk_idx = torch.topk(scores, args.num_topk, dim=-1, largest=True, sorted=False)
    topk_idx = topk_idx.to(deep_ep.topk_idx_t).contiguous()
    topk_weights = topk_weights.contiguous()
    x_bf16 = torch.randn((args.num_tokens, args.hidden), dtype=torch.bfloat16, device="cuda")
    x_fp8, sf = per_token_cast_to_fp8(x_bf16)
    return x_fp8.contiguous(), sf.contiguous(), topk_idx, topk_weights


def expected_dispatch_counts(topk_idx: torch.Tensor, num_ranks: int, num_experts: int) -> torch.Tensor:
    experts_per_rank = num_experts // num_ranks
    dst_rank = torch.div(topk_idx, experts_per_rank, rounding_mode="floor")
    counts = []
    for rank in range(num_ranks):
        counts.append((dst_rank == rank).any(dim=1).sum())
    return torch.stack(counts).to(torch.int32)


def run(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    assert args.num_ranks == 0 or args.num_ranks == num_ranks

    bytes_per_peer = args.bytes_per_peer
    pack_once = None
    pack_stats = None
    packed_counts = None
    if args.mode == "pack_copy":
        x_fp8, sf, topk_idx, topk_weights = make_dispatch_inputs(rank, args)
        token_bytes = deep_ep._C.l4_grouped_fp8_dispatch_token_bytes(
            args.hidden, sf.size(1), args.num_topk)
        bytes_per_peer = args.num_tokens * token_bytes
        packed = torch.empty((num_ranks * bytes_per_peer,), dtype=torch.uint8, device="cuda")
        packed_count = torch.empty((num_ranks,), dtype=torch.int32, device="cuda")

        def pack_once() -> None:
            deep_ep._C.l4_grouped_pack_fp8_dispatch_out(
                x_fp8, sf, topk_idx, topk_weights,
                packed, packed_count,
                rank, num_ranks, args.num_experts, args.num_sms)

        pack_once()
        torch.cuda.synchronize()
        expected_counts = expected_dispatch_counts(topk_idx, num_ranks, args.num_experts)
        assert torch.equal(packed_count.cpu(), expected_counts.cpu()), (
            rank, packed_count.cpu().tolist(), expected_counts.cpu().tolist())
        packed_counts = [int(x) for x in packed_count.cpu().tolist()]
        send = packed
        pack_stats = time_cuda(pack_once, args.warmups, args.iters)
    else:
        token_bytes = 0
        send = make_send_buffer(rank, num_ranks, bytes_per_peer, args.check)

    if bytes_per_peer % 32 != 0:
        raise ValueError("--bytes-per-peer must be 32-byte aligned")
    buffer_bytes = args.bytes_per_peer * num_ranks
    if args.mode == "pack_copy":
        buffer_bytes = bytes_per_peer * num_ranks
    buffer = deep_ep.ElasticBuffer(
        group,
        num_bytes=buffer_bytes,
        deterministic=False,
        allow_hybrid_mode=False,
        allow_multiple_reduction=True,
        prefer_overlap_with_compute=False,
        num_allocated_qps=args.num_allocated_qps,
        explicitly_destroy=True,
        num_gpu_timeout_secs=args.num_gpu_timeout_secs,
        num_cpu_timeout_secs=args.num_cpu_timeout_secs,
    )

    def p2p_once() -> torch.Tensor:
        return buffer.runtime.l4_p2p_all_to_all_fixed(send, bytes_per_peer)

    def pack_p2p_once() -> torch.Tensor:
        assert pack_once is not None
        pack_once()
        return p2p_once()

    out = p2p_once()
    torch.cuda.synchronize()
    if args.check and args.mode == "fixed_copy":
        check_output(out, rank, bytes_per_peer)
        torch.cuda.synchronize()

    copy_stats = time_cuda(p2p_once, args.warmups, args.iters)
    e2e_stats = time_cuda(pack_p2p_once, args.warmups, args.iters) if args.mode == "pack_copy" else None
    send_bytes = bytes_per_peer * num_ranks
    recv_bytes = bytes_per_peer * num_ranks
    local = {
        "backend": "deepep_l4_grouped_p2p_copy_engine",
        "mode": args.mode,
        "rank": rank,
        "rank_count": num_ranks,
        "bytes_per_peer": bytes_per_peer,
        "token_bytes": token_bytes,
        "packed_counts": packed_counts,
        "send_bytes": send_bytes,
        "recv_bytes": recv_bytes,
        "copy_median_us": copy_stats["median_s"] * 1e6,
        "copy_p90_us": copy_stats["p90_s"] * 1e6,
        "copy_min_us": copy_stats["min_s"] * 1e6,
        "copy_max_us": copy_stats["max_s"] * 1e6,
        "copy_send_gbs": send_bytes / 1e9 / copy_stats["median_s"],
        "copy_recv_gbs": recv_bytes / 1e9 / copy_stats["median_s"],
        "copy_bidirectional_gbs": (send_bytes + recv_bytes) / 1e9 / copy_stats["median_s"],
    }
    if pack_stats is not None:
        local.update({
            "pack_median_us": pack_stats["median_s"] * 1e6,
            "pack_p90_us": pack_stats["p90_s"] * 1e6,
            "pack_gbs": send_bytes / 1e9 / pack_stats["median_s"],
            "e2e_median_us": e2e_stats["median_s"] * 1e6,
            "e2e_p90_us": e2e_stats["p90_s"] * 1e6,
            "e2e_bidirectional_gbs": (send_bytes + recv_bytes) / 1e9 / e2e_stats["median_s"],
        })

    gathered = [None for _ in range(num_ranks)]
    dist.all_gather_object(gathered, local, group=group)
    if rank == 0:
        medians = [row["copy_median_us"] for row in gathered]
        bidir = [row["copy_bidirectional_gbs"] for row in gathered]
        result = {
            "backend": "deepep_l4_grouped_p2p_copy_engine",
            "mode": args.mode,
            "payload": args.payload,
            "rank_count": num_ranks,
            "bytes_per_peer": bytes_per_peer,
            "token_bytes": token_bytes,
            "warmups": args.warmups,
            "iters": args.iters,
            "copy_median_us_min": min(medians),
            "copy_median_us_max": max(medians),
            "copy_median_us_median": statistics.median(medians),
            "copy_bidirectional_gbs_min": min(bidir),
            "copy_bidirectional_gbs_max": max(bidir),
            "copy_bidirectional_gbs_median": statistics.median(bidir),
            "ranks": gathered,
        }
        if args.mode == "pack_copy":
            pack_medians = [row["pack_median_us"] for row in gathered]
            e2e_medians = [row["e2e_median_us"] for row in gathered]
            result.update({
                "pack_median_us_median": statistics.median(pack_medians),
                "pack_median_us_min": min(pack_medians),
                "pack_median_us_max": max(pack_medians),
                "e2e_median_us_median": statistics.median(e2e_medians),
                "e2e_median_us_min": min(e2e_medians),
                "e2e_median_us_max": max(e2e_medians),
            })
        print(json.dumps(result, sort_keys=True), flush=True)

    buffer.destroy()
    dist.barrier(group=group)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DeepEP L4 grouped P2P fixed-size all-to-all transport microbench")
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--num-ranks", type=int, default=0)
    parser.add_argument("--mode", type=str, default="fixed_copy", choices=["fixed_copy", "pack_copy"])
    parser.add_argument("--payload", type=str, default="dispatch_fp8",
                        choices=["dispatch_fp8", "combine_bf16", "custom"])
    parser.add_argument("--bytes-per-peer", type=int, default=56_141_280)
    parser.add_argument("--num-tokens", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--num-sms", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--num-allocated-qps", type=int, default=4)
    parser.add_argument("--num-gpu-timeout-secs", type=int, default=300)
    parser.add_argument("--num-cpu-timeout-secs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--unbalanced-ratio", type=float, default=1.0)
    parser.add_argument("--check", action="store_true")
    parsed = parser.parse_args()

    if parsed.payload == "combine_bf16" and parsed.bytes_per_peer == 56_141_280:
        parsed.bytes_per_peer = 107_484_160

    torch.multiprocessing.spawn(run, args=(parsed.num_processes, parsed), nprocs=parsed.num_processes)
