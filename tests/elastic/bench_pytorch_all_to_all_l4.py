import argparse
import json
import statistics
from typing import Dict, List

import torch
import torch.distributed as dist

from deep_ep.utils.envs import init_dist


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


def make_splits(num_ranks: int, rank: int, args: argparse.Namespace) -> torch.Tensor:
    # The baseline is intentionally byte-oriented. DeepEP's top-k routing can
    # duplicate one token to multiple ranks, so a token-balanced all-to-all would
    # undercount the transport work. Use measured EP payload bytes per peer.
    splits = torch.zeros((num_ranks,), dtype=torch.int64, device="cuda")
    if args.payload_mode == "balanced_all":
        splits.fill_(args.bytes_per_peer)
    elif args.payload_mode == "exclude_self":
        splits.fill_(args.bytes_per_peer)
        splits[rank] = 0
    elif args.payload_mode == "pair_offset":
        splits[(rank + args.pair_offset) % num_ranks] = args.bytes_per_peer
    else:
        raise ValueError(f"Unsupported payload mode: {args.payload_mode}")
    return splits


def run(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    assert args.num_ranks == 0 or args.num_ranks == num_ranks

    splits = make_splits(num_ranks, rank, args)
    recv_splits_tensor = torch.empty_like(splits)
    dist.all_to_all_single(recv_splits_tensor, splits, group=group)
    torch.cuda.synchronize()

    send_splits = [int(c) for c in splits.cpu().tolist()]
    recv_splits = [int(c) for c in recv_splits_tensor.cpu().tolist()]
    send_bytes = sum(send_splits)
    recv_bytes = sum(recv_splits)

    send = torch.empty((send_bytes,), dtype=torch.uint8, device="cuda")
    recv = torch.empty((recv_bytes,), dtype=torch.uint8, device="cuda")
    send.fill_(rank % 251)

    def all_to_all_payload() -> None:
        dist.all_to_all_single(recv, send, recv_splits, send_splits, group=group)

    all_to_all_payload()
    torch.cuda.synchronize()
    stats = time_cuda(all_to_all_payload, args.warmups, args.iters)

    local = {
        "rank": rank,
        "send_bytes": send_bytes,
        "recv_bytes": recv_bytes,
        "median_us": stats["median_s"] * 1e6,
        "p90_us": stats["p90_s"] * 1e6,
        "send_gbs": send_bytes / 1e9 / stats["median_s"] if send_bytes else 0.0,
        "recv_gbs": recv_bytes / 1e9 / stats["median_s"] if recv_bytes else 0.0,
        "bidirectional_gbs": (send_bytes + recv_bytes) / 1e9 / stats["median_s"],
    }
    gathered = [None for _ in range(num_ranks)]
    dist.all_gather_object(gathered, local, group=group)
    if rank == 0:
        medians = [row["median_us"] for row in gathered]
        bidir = [row["bidirectional_gbs"] for row in gathered]
        result = {
            "backend": "pytorch_nccl_all_to_all_single",
            "payload_mode": args.payload_mode,
            "rank_count": num_ranks,
            "bytes_per_peer": args.bytes_per_peer,
            "pair_offset": args.pair_offset,
            "warmups": args.warmups,
            "iters": args.iters,
            "median_us_min": min(medians),
            "median_us_max": max(medians),
            "median_us_median": statistics.median(medians),
            "bidirectional_gbs_min": min(bidir),
            "bidirectional_gbs_max": max(bidir),
            "bidirectional_gbs_median": statistics.median(bidir),
            "ranks": gathered,
        }
        print(json.dumps(result, sort_keys=True), flush=True)

    dist.barrier(group=group)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch/NCCL all_to_all_single payload baseline for L4 EP")
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--num-ranks", type=int, default=0)
    parser.add_argument("--bytes-per-peer", type=int, required=True)
    parser.add_argument("--payload-mode", type=str, default="balanced_all",
                        choices=["balanced_all", "exclude_self", "pair_offset"])
    parser.add_argument("--pair-offset", type=int, default=0)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parsed = parser.parse_args()
    torch.multiprocessing.spawn(run, args=(parsed.num_processes, parsed), nprocs=parsed.num_processes)
