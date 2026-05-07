import argparse
import json
import os
import statistics

import torch
import torch.distributed as dist

import deep_ep
from deep_ep.utils.envs import init_dist
from deep_ep.utils.gate import get_unbalanced_scores
from deep_ep.utils.math import count_bytes, per_token_cast_to_fp8
from deep_ep.utils.testing import bench_kineto


def time_cuda(fn, warmups: int, iters: int):
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
        "p90_s": sorted(times)[max(0, int(0.9 * len(times)) - 1)],
        "min_s": min(times),
        "max_s": max(times),
    }


def run(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    buffer = deep_ep.ElasticBuffer(
        group,
        num_max_tokens_per_rank=args.num_tokens,
        hidden=args.hidden,
        deterministic=False,
        allow_hybrid_mode=False,
        allow_multiple_reduction=True,
        prefer_overlap_with_compute=False,
        num_allocated_qps=max(args.num_allocated_qps, args.num_qps),
        explicitly_destroy=True,
        num_gpu_timeout_secs=args.num_gpu_timeout_secs,
        num_cpu_timeout_secs=args.num_cpu_timeout_secs,
    )

    torch.manual_seed(args.seed + rank)
    scores = get_unbalanced_scores(
        args.num_tokens, args.num_experts, buffer.num_ranks, args.num_topk, args.unbalanced_ratio, False)
    topk_weights, topk_idx = torch.topk(scores, args.num_topk, dim=-1, largest=True, sorted=False)
    topk_idx = topk_idx.to(deep_ep.topk_idx_t)

    x_bf16 = torch.randn((args.num_tokens, args.hidden), dtype=torch.bfloat16, device="cuda")
    x_dispatch = per_token_cast_to_fp8(x_bf16) if args.fp8_dispatch else x_bf16

    dispatch_args = dict(
        x=x_dispatch,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_sms=args.num_sms,
        num_qps=args.num_qps,
        num_max_tokens_per_rank=args.num_tokens,
        num_experts=args.num_experts,
        expert_alignment=args.expert_alignment,
        async_with_compute_stream=False,
        allocate_on_comm_stream=False,
        do_handle_copy=True,
        do_cpu_sync=bool(args.do_cpu_sync),
    )
    recv_x, recv_topk_idx, recv_topk_weights, handle, _ = buffer.dispatch(**dispatch_args)
    torch.cuda.synchronize()

    recv_payload = recv_x[0] if isinstance(recv_x, tuple) else recv_x
    combine_input = torch.randn((recv_payload.shape[0], args.hidden), dtype=torch.bfloat16, device="cuda")
    combine_args = dict(
        x=combine_input,
        topk_weights=recv_topk_weights,
        bias=None,
        handle=handle,
        num_sms=args.num_sms,
        num_qps=args.num_qps,
        async_with_compute_stream=False,
        allocate_on_comm_stream=False,
    )
    combined_x, combined_topk_weights, _ = buffer.combine(**combine_args)
    torch.cuda.synchronize()
    assert combined_x.shape == (args.num_tokens, args.hidden)
    assert combined_topk_weights.shape == topk_weights.shape

    dispatch_stats = time_cuda(lambda: buffer.dispatch(**dispatch_args), args.warmups, args.iters)
    combine_stats = time_cuda(lambda: buffer.combine(**combine_args), args.warmups, args.iters)

    profile = {}
    if args.profile:
        trace_dir = args.trace_dir or None
        if trace_dir and rank == 0:
            os.makedirs(trace_dir, exist_ok=True)
        dist.barrier()
        dispatch_trace = None if not trace_dir else os.path.join(trace_dir, f"dispatch_rank{rank}.json")
        combine_trace = None if not trace_dir else os.path.join(trace_dir, f"combine_rank{rank}.json")
        dispatch_kernel_s, copy_kernel_s = bench_kineto(
            lambda: buffer.dispatch(**dispatch_args),
            ("dispatch_impl", "dispatch_copy_epilogue_impl"),
            num_tests=args.profile_iters,
            trace_path=dispatch_trace,
            barrier_comm_profiling=True,
            barrier=buffer.barrier,
        )
        combine_kernel_s = bench_kineto(
            lambda: buffer.combine(**combine_args),
            "combine_impl",
            num_tests=args.profile_iters,
            trace_path=combine_trace,
            barrier_comm_profiling=True,
            barrier=buffer.barrier,
        )
        profile = {
            "dispatch_kernel_us": dispatch_kernel_s * 1e6,
            "dispatch_copy_epilogue_us": copy_kernel_s * 1e6,
            "combine_kernel_us": combine_kernel_s * 1e6,
        }

    dispatch_bytes = count_bytes(recv_x, recv_topk_idx, recv_topk_weights)
    combine_bytes = count_bytes(combine_input)
    result = {
        "rank_count": num_ranks,
        "num_tokens": args.num_tokens,
        "hidden": args.hidden,
        "num_topk": args.num_topk,
        "num_experts": args.num_experts,
        "num_sms": args.num_sms,
        "num_qps": args.num_qps,
        "dispatch_dtype": "fp8_e4m3" if args.fp8_dispatch else "bf16",
        "combine_dtype": "bf16",
        "dispatch_bytes": dispatch_bytes,
        "combine_bytes": combine_bytes,
        "dispatch_median_us": dispatch_stats["median_s"] * 1e6,
        "dispatch_p90_us": dispatch_stats["p90_s"] * 1e6,
        "dispatch_gbs": dispatch_bytes / 1e9 / dispatch_stats["median_s"],
        "combine_median_us": combine_stats["median_s"] * 1e6,
        "combine_p90_us": combine_stats["p90_s"] * 1e6,
        "combine_gbs": combine_bytes / 1e9 / combine_stats["median_s"],
        **profile,
    }
    if rank == 0:
        print(json.dumps(result, sort_keys=True), flush=True)

    buffer.destroy()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Elastic/JIT EP kernels on the 4xL4 PCIe target")
    parser.add_argument("--num-processes", type=int, default=4)
    parser.add_argument("--num-tokens", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--num-sms", type=int, default=10)
    parser.add_argument("--num-qps", type=int, default=4)
    parser.add_argument("--num-allocated-qps", type=int, default=4)
    parser.add_argument("--expert-alignment", type=int, default=128)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-iters", type=int, default=5)
    parser.add_argument("--trace-dir", type=str, default="")
    parser.add_argument("--do-cpu-sync", type=int, default=1)
    parser.add_argument("--num-gpu-timeout-secs", type=int, default=300)
    parser.add_argument("--num-cpu-timeout-secs", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--unbalanced-ratio", type=float, default=1.0)
    dispatch_dtype = parser.add_mutually_exclusive_group()
    dispatch_dtype.add_argument("--fp8-dispatch", dest="fp8_dispatch", action="store_true")
    dispatch_dtype.add_argument("--bf16-dispatch", dest="fp8_dispatch", action="store_false")
    parser.set_defaults(fp8_dispatch=True)
    parsed = parser.parse_args()
    torch.multiprocessing.spawn(run, args=(parsed.num_processes, parsed), nprocs=parsed.num_processes)
