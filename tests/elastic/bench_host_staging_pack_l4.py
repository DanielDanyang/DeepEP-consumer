import argparse
import json
import statistics
from typing import Dict, List

import torch
import torch.multiprocessing as mp

import deep_ep
from deep_ep.utils.gate import get_unbalanced_scores
from deep_ep.utils.math import count_bytes, per_token_cast_to_fp8


def percentile(values: List[float], q: float) -> float:
    values = sorted(values)
    return values[max(0, min(len(values) - 1, int(q * len(values)) - 1))]


def parse_csv_ints(value: str) -> List[int]:
    return [int(item) for item in value.split(",") if item.strip()]


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


def get_remote_mask(topk_idx: torch.Tensor,
                    rank_idx: int,
                    num_ranks: int,
                    num_scaleup_ranks: int,
                    num_experts: int) -> torch.Tensor:
    num_scaleout_ranks = num_ranks // num_scaleup_ranks
    local_scaleout_rank_idx = rank_idx // num_scaleup_ranks
    num_experts_per_scaleout_rank = num_experts // num_scaleout_ranks
    dst_scaleout = torch.div(topk_idx, num_experts_per_scaleout_rank, rounding_mode="floor")
    return ((topk_idx >= 0) & (dst_scaleout != local_scaleout_rank_idx)).any(dim=1)


def worker(local_rank: int, devices: List[int], args: argparse.Namespace, queue: mp.Queue) -> None:
    device = devices[local_rank]
    rank_idx = args.node_rank * len(devices) + local_rank
    torch.cuda.set_device(device)
    torch.set_default_device("cuda")

    torch.manual_seed(args.seed + rank_idx)
    scores = get_unbalanced_scores(
        args.num_tokens, args.num_experts, args.num_ranks, args.num_topk, args.unbalanced_ratio, False)
    topk_weights, topk_idx = torch.topk(scores, args.num_topk, dim=-1, largest=True, sorted=False)
    topk_idx = topk_idx.to(deep_ep.topk_idx_t).contiguous()

    x_bf16 = torch.randn((args.num_tokens, args.hidden), dtype=torch.bfloat16, device="cuda")
    x_fp8, sf = per_token_cast_to_fp8(x_bf16)
    x_fp8 = x_fp8.contiguous()
    sf = sf.contiguous()
    topk_weights = topk_weights.contiguous()

    def pack():
        return deep_ep._C.host_staging_pack_fp8_dispatch(
            x_fp8, sf, topk_idx, topk_weights,
            rank_idx, args.num_ranks, args.num_scaleup_ranks, args.num_experts, args.num_sms)

    packed_x, packed_sf, packed_topk_idx, packed_topk_weights, packed_src_token_idx, packed_count = pack()
    torch.cuda.synchronize()

    remote_mask = get_remote_mask(topk_idx, rank_idx, args.num_ranks, args.num_scaleup_ranks, args.num_experts)
    expected_count = int(remote_mask.sum().item())
    actual_count = int(packed_count.cpu().item())
    assert actual_count == expected_count, (actual_count, expected_count)

    packed_src = packed_src_token_idx[:actual_count].long()
    expected_src = remote_mask.nonzero(as_tuple=False).flatten()
    assert torch.equal(torch.sort(packed_src).values, torch.sort(expected_src).values)
    if args.check_payload and actual_count > 0:
        assert torch.equal(packed_x[:actual_count], x_fp8[packed_src])
        assert torch.equal(packed_sf[:actual_count], sf[packed_src])
        assert torch.equal(packed_topk_idx[:actual_count], topk_idx[packed_src])
        assert torch.equal(packed_topk_weights[:actual_count], topk_weights[packed_src])

    stats = time_cuda(pack, args.warmups, args.iters)
    packed_bytes = count_bytes(
        packed_x[:actual_count],
        packed_sf[:actual_count],
        packed_topk_idx[:actual_count],
        packed_topk_weights[:actual_count],
        packed_src_token_idx[:actual_count])
    queue.put({
        "device": device,
        "rank_idx": rank_idx,
        "num_tokens": args.num_tokens,
        "hidden": args.hidden,
        "num_topk": args.num_topk,
        "num_experts": args.num_experts,
        "num_ranks": args.num_ranks,
        "num_scaleup_ranks": args.num_scaleup_ranks,
        "remote_tokens": actual_count,
        "remote_ratio": actual_count / args.num_tokens,
        "packed_bytes": packed_bytes,
        "pack_median_us": stats["median_s"] * 1e6,
        "pack_p90_us": stats["p90_s"] * 1e6,
        "pack_gbs": packed_bytes / 1e9 / stats["median_s"] if stats["median_s"] > 0 else 0,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the SM89 host-staging FP8 dispatch pack kernel")
    parser.add_argument("--devices", type=str, default="0,1,2,3")
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--num-ranks", type=int, default=8)
    parser.add_argument("--num-scaleup-ranks", type=int, default=4)
    parser.add_argument("--num-tokens", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--num-sms", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--unbalanced-ratio", type=float, default=1.0)
    parser.add_argument("--check-payload", action="store_true")
    args = parser.parse_args()

    devices = parse_csv_ints(args.devices)
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    mp.spawn(worker, args=(devices, args, queue), nprocs=len(devices))
    rows = [queue.get() for _ in devices]
    rows.sort(key=lambda item: item["rank_idx"])
    for row in rows:
        print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
