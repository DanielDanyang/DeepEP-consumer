import argparse
import json
import statistics
from typing import Dict, List, Tuple

import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import init_dist
from deep_ep.utils.gate import get_unbalanced_scores
from deep_ep.utils.math import count_bytes, per_token_cast_to_fp8


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


def make_inputs(rank: int, num_ranks: int, args: argparse.Namespace):
    torch.manual_seed(args.seed + rank)
    scores = get_unbalanced_scores(
        args.num_tokens, args.num_experts, num_ranks, args.num_topk, args.unbalanced_ratio, False)
    topk_weights, topk_idx = torch.topk(scores, args.num_topk, dim=-1, largest=True, sorted=False)
    topk_idx = topk_idx.to(deep_ep.topk_idx_t).contiguous()
    topk_weights = topk_weights.contiguous()
    x_bf16 = torch.randn((args.num_tokens, args.hidden), dtype=torch.bfloat16, device="cuda")
    x_fp8, sf = per_token_cast_to_fp8(x_bf16)
    src = torch.arange(args.num_tokens, dtype=torch.int32, device="cuda") + rank * args.num_tokens
    return x_fp8.contiguous(), sf.contiguous(), topk_idx, topk_weights, src


def pack_by_rank(x_fp8: torch.Tensor,
                 sf: torch.Tensor,
                 topk_idx: torch.Tensor,
                 topk_weights: torch.Tensor,
                 src: torch.Tensor,
                 num_ranks: int,
                 num_experts: int) -> Tuple[Tuple[torch.Tensor, ...], List[int]]:
    experts_per_rank = num_experts // num_ranks
    dst_rank = torch.div(topk_idx, experts_per_rank, rounding_mode="floor")
    xs, sfs, idxs, weights, srcs, counts = [], [], [], [], [], []
    for rank in range(num_ranks):
        mask = (dst_rank == rank).any(dim=1)
        token_idx = mask.nonzero(as_tuple=False).flatten()
        counts.append(int(token_idx.numel()))
        xs.append(x_fp8.view(torch.uint8)[token_idx])
        sfs.append(sf[token_idx])
        idxs.append(topk_idx[token_idx])
        weights.append(topk_weights[token_idx])
        srcs.append(src[token_idx])
    return (
        torch.cat(xs, dim=0) if xs else x_fp8.view(torch.uint8)[:0],
        torch.cat(sfs, dim=0) if sfs else sf[:0],
        torch.cat(idxs, dim=0) if idxs else topk_idx[:0],
        torch.cat(weights, dim=0) if weights else topk_weights[:0],
        torch.cat(srcs, dim=0) if srcs else src[:0],
    ), counts


def exchange_counts(send_counts: List[int], group) -> List[int]:
    send = torch.tensor(send_counts, dtype=torch.int64, device="cuda")
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=group)
    return [int(x) for x in recv.cpu().tolist()]


def all_to_all_tensor(t: torch.Tensor,
                     send_counts: List[int],
                     recv_counts: List[int],
                     elems_per_token: int,
                     group) -> torch.Tensor:
    send_splits = [count * elems_per_token for count in send_counts]
    recv_splits = [count * elems_per_token for count in recv_counts]
    out = torch.empty((sum(recv_splits),), dtype=t.dtype, device=t.device)
    dist.all_to_all_single(out, t.reshape(-1), recv_splits, send_splits, group=group)
    return out.reshape((sum(recv_counts), elems_per_token))


def unpack_ep(recv_topk_idx: torch.Tensor,
              recv_topk_weights: torch.Tensor,
              recv_src: torch.Tensor,
              rank: int,
              num_ranks: int,
              num_experts: int) -> Tuple[torch.Tensor, torch.Tensor]:
    experts_per_rank = num_experts // num_ranks
    expert_start = rank * experts_per_rank
    expert_end = expert_start + experts_per_rank
    local_topk_idx = recv_topk_idx.clone()
    in_range = (local_topk_idx >= expert_start) & (local_topk_idx < expert_end)
    local_topk_idx = torch.where(in_range, local_topk_idx - expert_start, torch.full_like(local_topk_idx, -1))

    # Match DeepEP non-expanded metadata shape enough for correctness and
    # combine-readiness: source global token index plus the first matching
    # top-k lane. The full DeepEP handle also carries rank-prefix tensors.
    first_lane = torch.argmax(in_range.to(torch.int32), dim=1).to(torch.int32)
    src_metadata = torch.empty((recv_topk_idx.size(0), recv_topk_idx.size(1) + 2), dtype=torch.int32, device="cuda")
    src_metadata[:, 0] = recv_src.to(torch.int32)
    src_metadata[:, 1] = first_lane
    src_metadata[:, 2:] = -1
    return local_topk_idx, src_metadata


def run(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    assert args.num_ranks == 0 or args.num_ranks == num_ranks
    assert args.num_experts % num_ranks == 0

    x_fp8, sf, topk_idx, topk_weights, src = make_inputs(rank, num_ranks, args)

    def ep_once():
        packed, send_counts = pack_by_rank(x_fp8, sf, topk_idx, topk_weights, src, num_ranks, args.num_experts)
        recv_counts = exchange_counts(send_counts, group)
        recv_x = all_to_all_tensor(packed[0], send_counts, recv_counts, args.hidden, group)
        recv_sf = all_to_all_tensor(packed[1], send_counts, recv_counts, sf.size(1), group)
        recv_topk_idx = all_to_all_tensor(packed[2], send_counts, recv_counts, args.num_topk, group)
        recv_topk_weights = all_to_all_tensor(packed[3], send_counts, recv_counts, args.num_topk, group)
        recv_src = all_to_all_tensor(packed[4].view(-1, 1), send_counts, recv_counts, 1, group).view(-1)
        local_topk_idx, src_metadata = unpack_ep(recv_topk_idx, recv_topk_weights, recv_src, rank, num_ranks, args.num_experts)
        return recv_x, recv_sf, local_topk_idx, recv_topk_weights, src_metadata, send_counts, recv_counts

    recv_x, recv_sf, recv_topk_idx, recv_topk_weights, src_metadata, send_counts, recv_counts = ep_once()
    torch.cuda.synchronize()
    assert recv_x.size(0) == sum(recv_counts)
    assert recv_sf.size(0) == sum(recv_counts)
    assert recv_topk_idx.shape == (sum(recv_counts), args.num_topk)
    assert recv_topk_weights.shape == (sum(recv_counts), args.num_topk)
    assert src_metadata.shape == (sum(recv_counts), args.num_topk + 2)

    def payload_once():
        recv_x_ = all_to_all_tensor(packed_inputs[0], send_counts, recv_counts, args.hidden, group)
        recv_sf_ = all_to_all_tensor(packed_inputs[1], send_counts, recv_counts, sf.size(1), group)
        recv_topk_idx_ = all_to_all_tensor(packed_inputs[2], send_counts, recv_counts, args.num_topk, group)
        recv_topk_weights_ = all_to_all_tensor(packed_inputs[3], send_counts, recv_counts, args.num_topk, group)
        recv_src_ = all_to_all_tensor(packed_inputs[4].view(-1, 1), send_counts, recv_counts, 1, group).view(-1)
        return recv_x_, recv_sf_, recv_topk_idx_, recv_topk_weights_, recv_src_

    packed_inputs, send_counts = pack_by_rank(x_fp8, sf, topk_idx, topk_weights, src, num_ranks, args.num_experts)
    recv_counts = exchange_counts(send_counts, group)
    recv_payload = payload_once()
    torch.cuda.synchronize()

    stats = time_cuda(ep_once, args.warmups, args.iters)
    pack_stats = time_cuda(
        lambda: pack_by_rank(x_fp8, sf, topk_idx, topk_weights, src, num_ranks, args.num_experts),
        args.warmups, args.iters)
    count_stats = time_cuda(lambda: exchange_counts(send_counts, group), args.warmups, args.iters)
    payload_stats = time_cuda(payload_once, args.warmups, args.iters)
    unpack_stats = time_cuda(
        lambda: unpack_ep(recv_payload[2], recv_payload[3], recv_payload[4], rank, num_ranks, args.num_experts),
        args.warmups, args.iters)
    local_bytes = count_bytes(recv_x, recv_sf, recv_topk_idx, recv_topk_weights, src_metadata)
    local = {
        "backend": "pytorch_ep_pack_all_to_all_unpack",
        "rank": rank,
        "rank_count": num_ranks,
        "send_counts": send_counts,
        "recv_counts": recv_counts,
        "recv_tokens": sum(recv_counts),
        "recv_bytes": local_bytes,
        "median_us": stats["median_s"] * 1e6,
        "p90_us": stats["p90_s"] * 1e6,
        "recv_gbs": local_bytes / 1e9 / stats["median_s"],
        "pack_median_us": pack_stats["median_s"] * 1e6,
        "count_exchange_median_us": count_stats["median_s"] * 1e6,
        "payload_all_to_all_median_us": payload_stats["median_s"] * 1e6,
        "unpack_median_us": unpack_stats["median_s"] * 1e6,
    }
    gathered = [None for _ in range(num_ranks)]
    dist.all_gather_object(gathered, local, group=group)
    if rank == 0:
        medians = [row["median_us"] for row in gathered]
        recv_gbs = [row["recv_gbs"] for row in gathered]
        pack_medians = [row["pack_median_us"] for row in gathered]
        count_medians = [row["count_exchange_median_us"] for row in gathered]
        payload_medians = [row["payload_all_to_all_median_us"] for row in gathered]
        unpack_medians = [row["unpack_median_us"] for row in gathered]
        result = {
            "backend": "pytorch_ep_pack_all_to_all_unpack",
            "rank_count": num_ranks,
            "num_tokens": args.num_tokens,
            "hidden": args.hidden,
            "num_topk": args.num_topk,
            "num_experts": args.num_experts,
            "warmups": args.warmups,
            "iters": args.iters,
            "median_us_min": min(medians),
            "median_us_max": max(medians),
            "median_us_median": statistics.median(medians),
            "recv_gbs_min": min(recv_gbs),
            "recv_gbs_max": max(recv_gbs),
            "recv_gbs_median": statistics.median(recv_gbs),
            "pack_median_us_median": statistics.median(pack_medians),
            "count_exchange_median_us_median": statistics.median(count_medians),
            "payload_all_to_all_median_us_median": statistics.median(payload_medians),
            "unpack_median_us_median": statistics.median(unpack_medians),
            "ranks": gathered,
        }
        print(json.dumps(result, sort_keys=True), flush=True)

    dist.barrier(group=group)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Full PyTorch EP pack/all_to_all/unpack baseline for L4")
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--num-ranks", type=int, default=0)
    parser.add_argument("--num-tokens", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--unbalanced-ratio", type=float, default=1.0)
    parsed = parser.parse_args()
    torch.multiprocessing.spawn(run, args=(parsed.num_processes, parsed), nprocs=parsed.num_processes)
