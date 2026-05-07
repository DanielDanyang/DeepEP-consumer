import argparse
import json
import statistics
import time
from typing import Dict, List

import torch
import torch.multiprocessing as mp


def percentile(values: List[float], q: float) -> float:
    values = sorted(values)
    return values[max(0, min(len(values) - 1, int(q * len(values)) - 1))]


def parse_csv_ints(value: str) -> List[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def schedule_copy(host: torch.Tensor,
                  gpu: torch.Tensor,
                  streams: List[torch.cuda.Stream],
                  direction: str) -> None:
    num_streams = len(streams)
    chunk = host.numel() // num_streams
    for idx, stream in enumerate(streams):
        start = idx * chunk
        end = host.numel() if idx == num_streams - 1 else (idx + 1) * chunk
        with torch.cuda.stream(stream):
            if direction == "h2d":
                gpu[start:end].copy_(host[start:end], non_blocking=True)
            elif direction == "d2h":
                host[start:end].copy_(gpu[start:end], non_blocking=True)
            else:
                raise ValueError(f"Unknown direction: {direction}")


def time_copy(host: torch.Tensor,
              gpu: torch.Tensor,
              streams: List[torch.cuda.Stream],
              direction: str,
              warmups: int,
              iters: int) -> Dict[str, float]:
    torch.cuda.synchronize()
    for _ in range(warmups):
        schedule_copy(host, gpu, streams, direction)
        torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        schedule_copy(host, gpu, streams, direction)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)

    return {
        "median_s": statistics.median(times),
        "p90_s": percentile(times, 0.9),
        "min_s": min(times),
        "max_s": max(times),
    }


def worker(local_rank: int, devices: List[int], args: argparse.Namespace, queue: mp.Queue) -> None:
    device = devices[local_rank]
    torch.cuda.set_device(device)

    results = []
    for size_mb in args.sizes_mb:
        num_bytes = size_mb * 1024 * 1024
        host = torch.empty((num_bytes,), dtype=torch.uint8, pin_memory=True)
        gpu = torch.empty((num_bytes,), dtype=torch.uint8, device="cuda")
        host.fill_(local_rank + 1)
        gpu.fill_(local_rank + 3)
        for num_streams in args.streams:
            streams = [torch.cuda.Stream(device=device) for _ in range(num_streams)]
            for direction in args.directions:
                stats = time_copy(host, gpu, streams, direction, args.warmups, args.iters)
                results.append({
                    "device": device,
                    "direction": direction,
                    "size_mb": size_mb,
                    "streams": num_streams,
                    "median_us": stats["median_s"] * 1e6,
                    "p90_us": stats["p90_s"] * 1e6,
                    "gbs": num_bytes / 1e9 / stats["median_s"],
                })
    queue.put(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure pinned host PCIe copy bandwidth for L4 host staging")
    parser.add_argument("--devices", type=str, default="0,1,2,3")
    parser.add_argument("--sizes-mb", type=int, nargs="+", default=[16, 64, 128, 256])
    parser.add_argument("--streams", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--directions", type=str, nargs="+", default=["h2d", "d2h"], choices=["h2d", "d2h"])
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iters", type=int, default=8)
    args = parser.parse_args()

    devices = parse_csv_ints(args.devices)
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    mp.spawn(worker, args=(devices, args, queue), nprocs=len(devices))
    all_results = []
    for _ in devices:
        all_results.extend(queue.get())
    all_results.sort(key=lambda item: (item["device"], item["size_mb"], item["streams"], item["direction"]))
    for row in all_results:
        print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
