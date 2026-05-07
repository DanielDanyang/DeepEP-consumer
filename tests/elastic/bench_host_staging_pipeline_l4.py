import argparse
import json
import statistics
import time
from typing import Dict, List, Sequence, Tuple

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


def time_host(fn, warmups: int, iters: int) -> Dict[str, float]:
    torch.cuda.synchronize()
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return {
        "median_s": statistics.median(times),
        "p90_s": percentile(times, 0.9),
        "min_s": min(times),
        "max_s": max(times),
    }


def byte_view(tensor: torch.Tensor, num_bytes: int) -> torch.Tensor:
    return tensor.view(torch.uint8).reshape(-1)[:num_bytes]


def schedule_copy_chunks(src: torch.Tensor,
                         dst: torch.Tensor,
                         streams: Sequence[torch.cuda.Stream],
                         wait_event: torch.cuda.Event | None = None) -> None:
    num_streams = len(streams)
    chunk = src.numel() // num_streams
    for idx, stream in enumerate(streams):
        start = idx * chunk
        end = src.numel() if idx == num_streams - 1 else (idx + 1) * chunk
        with torch.cuda.stream(stream):
            if wait_event is not None:
                stream.wait_event(wait_event)
            dst[start:end].copy_(src[start:end], non_blocking=True)


def schedule_copy_many(pairs: Sequence[Tuple[torch.Tensor, torch.Tensor]],
                       streams: Sequence[torch.cuda.Stream],
                       wait_event: torch.cuda.Event | None = None) -> None:
    for src, dst in pairs:
        schedule_copy_chunks(src, dst, streams, wait_event)


def make_packed_byte_views(outputs: Tuple[torch.Tensor, ...],
                           actual_count: int,
                           hidden: int,
                           num_sf_packs: int,
                           num_topk: int,
                           topk_idx_elem_size: int) -> List[torch.Tensor]:
    packed_x, packed_sf, packed_topk_idx, packed_topk_weights, packed_src_token_idx, _ = outputs
    return [
        byte_view(packed_x, actual_count * hidden),
        byte_view(packed_sf, actual_count * num_sf_packs * packed_sf.element_size()),
        byte_view(packed_topk_idx, actual_count * num_topk * topk_idx_elem_size),
        byte_view(packed_topk_weights, actual_count * num_topk * packed_topk_weights.element_size()),
        byte_view(packed_src_token_idx, actual_count * packed_src_token_idx.element_size()),
    ]


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

    reusable_outputs = (
        torch.empty_like(x_fp8),
        torch.empty_like(sf),
        torch.empty_like(topk_idx),
        torch.empty_like(topk_weights),
        torch.empty((args.num_tokens,), dtype=torch.int32, device="cuda"),
        torch.empty((1,), dtype=torch.int32, device="cuda"),
    )

    def pack() -> Tuple[torch.Tensor, ...]:
        deep_ep._C.host_staging_pack_fp8_dispatch_out(
            x_fp8, sf, topk_idx, topk_weights,
            reusable_outputs[0], reusable_outputs[1], reusable_outputs[2],
            reusable_outputs[3], reusable_outputs[4], reusable_outputs[5],
            rank_idx, args.num_ranks, args.num_scaleup_ranks, args.num_experts, args.num_sms)
        return reusable_outputs

    pack()
    torch.cuda.synchronize()
    actual_count = int(reusable_outputs[5].cpu().item())
    topk_idx_elem_size = topk_idx.element_size()
    packed_views = make_packed_byte_views(
        reusable_outputs, actual_count, args.hidden, sf.size(1), args.num_topk, topk_idx_elem_size)
    host_views = [
        torch.empty((view.numel(),), dtype=torch.uint8, device="cpu", pin_memory=True)
        for view in packed_views
    ]
    h2d_views = [torch.empty_like(view) for view in packed_views]
    copy_pairs_d2h = list(zip(packed_views, host_views))
    copy_pairs_h2d = list(zip(host_views, h2d_views))
    packed_bytes = sum(view.numel() for view in packed_views)

    rows = []
    for num_copy_streams in args.copy_streams:
        copy_streams = [torch.cuda.Stream(device=device) for _ in range(num_copy_streams)]

        def d2h_only() -> None:
            schedule_copy_many(copy_pairs_d2h, copy_streams)

        def h2d_only() -> None:
            schedule_copy_many(copy_pairs_h2d, copy_streams)

        def pack_then_d2h() -> None:
            pack()
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream())
            schedule_copy_many(copy_pairs_d2h, copy_streams, event)

        def pack_then_d2h_then_h2d() -> None:
            pack_then_d2h()
            torch.cuda.synchronize()
            h2d_only()

        stage_stats = {
            "pack": time_host(pack, args.warmups, args.iters),
            "d2h": time_host(d2h_only, args.warmups, args.iters),
            "h2d": time_host(h2d_only, args.warmups, args.iters),
            "pack_d2h": time_host(pack_then_d2h, args.warmups, args.iters),
            "pack_d2h_h2d": time_host(pack_then_d2h_then_h2d, args.warmups, args.iters),
        }
        if args.check_h2d:
            h2d_only()
            torch.cuda.synchronize()
            for src, dst in zip(packed_views, h2d_views):
                assert torch.equal(src, dst)

        row = {
            "device": device,
            "rank_idx": rank_idx,
            "num_tokens": args.num_tokens,
            "hidden": args.hidden,
            "num_topk": args.num_topk,
            "num_experts": args.num_experts,
            "remote_tokens": actual_count,
            "remote_ratio": actual_count / args.num_tokens,
            "packed_bytes": packed_bytes,
            "copy_streams": num_copy_streams,
        }
        for stage, stats in stage_stats.items():
            row[f"{stage}_median_us"] = stats["median_s"] * 1e6
            row[f"{stage}_p90_us"] = stats["p90_s"] * 1e6
            row[f"{stage}_gbs"] = packed_bytes / 1e9 / stats["median_s"] if stats["median_s"] > 0 else 0
        rows.append(row)
    queue.put(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SM89 host-staging pack plus pinned D2H/H2D stages")
    parser.add_argument("--devices", type=str, default="0,1,2,3")
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--num-ranks", type=int, default=8)
    parser.add_argument("--num-scaleup-ranks", type=int, default=4)
    parser.add_argument("--num-tokens", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--num-sms", type=int, default=10)
    parser.add_argument("--copy-streams", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--unbalanced-ratio", type=float, default=1.0)
    parser.add_argument("--check-h2d", action="store_true")
    args = parser.parse_args()

    devices = parse_csv_ints(args.devices)
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    mp.spawn(worker, args=(devices, args, queue), nprocs=len(devices))
    rows = []
    for _ in devices:
        rows.extend(queue.get())
    rows.sort(key=lambda item: (item["rank_idx"], item["copy_streams"]))
    for row in rows:
        print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
