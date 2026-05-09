import argparse
import hashlib
import json
import math
import socket
import statistics
import struct
import time
from typing import Dict, List, Sequence, Tuple

import torch
import torch.multiprocessing as mp

import deep_ep


def parse_csv_ints(value: str) -> List[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def percentile(values: List[float], q: float) -> float:
    values = sorted(values)
    return values[max(0, min(len(values) - 1, math.ceil(q * len(values)) - 1))]


def byte_view(tensor: torch.Tensor, num_bytes: int) -> torch.Tensor:
    return tensor.view(torch.uint8).reshape(-1)[:num_bytes]


def tensor_digest(tensor: torch.Tensor) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(memoryview(tensor.numpy()))
    return h.hexdigest()


def send_json(sock: socket.socket, payload: Dict[str, object]) -> None:
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    sock.sendall(struct.pack("!Q", len(blob)))
    sock.sendall(blob)


def recv_json(sock: socket.socket) -> Dict[str, object]:
    header = recv_exact(sock, 8)
    size = struct.unpack("!Q", header)[0]
    return json.loads(recv_exact(sock, size).decode("utf-8"))


def recv_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray(size)
    view = memoryview(data)
    offset = 0
    while offset < size:
        n = sock.recv_into(view[offset:])
        if n == 0:
            raise RuntimeError("Socket closed while receiving")
        offset += n
    return data


def recv_tensor(sock: socket.socket, tensor: torch.Tensor) -> None:
    view = memoryview(tensor.numpy())
    offset = 0
    while offset < tensor.numel():
        n = sock.recv_into(view[offset:])
        if n == 0:
            raise RuntimeError("Socket closed while receiving tensor")
        offset += n


def send_tensors(sock: socket.socket, tensors: Sequence[torch.Tensor]) -> None:
    for tensor in tensors:
        if tensor.numel() == 0:
            continue
        sock.sendall(memoryview(tensor.numpy()))


def exchange_tensors(sock: socket.socket,
                     node_rank: int,
                     host_send: Sequence[torch.Tensor],
                     host_recv: Sequence[torch.Tensor]) -> None:
    # Keep the transfer order deterministic to avoid deadlock with blocking
    # Python sockets. This is only a bring-up transport; the performance path
    # should replace it with persistent ibverbs QPs and full-duplex posting.
    if node_rank == 0:
        for tensor in host_recv:
            recv_tensor(sock, tensor)
        send_tensors(sock, host_send)
    else:
        send_tensors(sock, host_send)
        for tensor in host_recv:
            recv_tensor(sock, tensor)


def summarize_us(values_s: List[float]) -> Dict[str, float]:
    return {
        "median_us": statistics.median(values_s) * 1e6,
        "p90_us": percentile(values_s, 0.9) * 1e6,
        "min_us": min(values_s) * 1e6,
        "max_us": max(values_s) * 1e6,
    }


def connect_with_retry(host: str, port: int, timeout_s: float) -> socket.socket:
    deadline = time.perf_counter() + timeout_s
    last_error = None
    while time.perf_counter() < deadline:
        try:
            sock = socket.create_connection((host, port), timeout=5.0)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.settimeout(None)
            return sock
        except OSError as exc:
            last_error = exc
            time.sleep(0.2)
    raise RuntimeError(f"Could not connect to {host}:{port}: {last_error}")


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


def make_bf16_combine_byte_views(outputs: Tuple[torch.Tensor, ...],
                                 actual_count: int,
                                 hidden: int) -> List[torch.Tensor]:
    packed_x, packed_src_token_idx, _ = outputs
    return [
        byte_view(packed_x, actual_count * hidden * packed_x.element_size()),
        byte_view(packed_src_token_idx, actual_count * packed_src_token_idx.element_size()),
    ]


def copy_many(srcs: Sequence[torch.Tensor],
              dsts: Sequence[torch.Tensor],
              streams: Sequence[torch.cuda.Stream]) -> None:
    for src, dst in zip(srcs, dsts):
        if src.numel() == 0:
            continue
        chunk = src.numel() // len(streams)
        for idx, stream in enumerate(streams):
            start = idx * chunk
            end = src.numel() if idx == len(streams) - 1 else (idx + 1) * chunk
            with torch.cuda.stream(stream):
                dst[start:end].copy_(src[start:end], non_blocking=True)


def worker(local_rank: int, devices: List[int], args: argparse.Namespace, queue: mp.Queue) -> None:
    device = devices[local_rank]
    rank_idx = args.node_rank * len(devices) + local_rank
    peer_rank_idx = (1 - args.node_rank) * len(devices) + local_rank
    torch.cuda.set_device(device)
    torch.set_default_device("cuda")
    torch.manual_seed(args.seed + rank_idx)

    scores = torch.rand((args.num_tokens, args.num_experts), dtype=torch.float32, device="cuda")
    topk_weights, topk_idx = torch.topk(scores, args.num_topk, dim=-1, largest=True, sorted=False)
    topk_idx = topk_idx.to(deep_ep.topk_idx_t).contiguous()
    topk_weights = topk_weights.contiguous()

    # This smoke measures the explicit host-staging payload path, so avoid pulling
    # torch.compile FP8 casting into the critical path. Any byte is a valid E4M3
    # storage value for copy/transport validation.
    if args.payload_mode == "dispatch_fp8":
        x_fp8 = torch.randint(
            0, 256, (args.num_tokens, args.hidden), dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
        sf = torch.rand((args.num_tokens, (args.hidden + 127) // 128), dtype=torch.float32, device="cuda")
        outputs = (
            torch.empty_like(x_fp8),
            torch.empty_like(sf),
            torch.empty_like(topk_idx),
            torch.empty_like(topk_weights),
            torch.empty((args.num_tokens,), dtype=torch.int32, device="cuda"),
            torch.empty((1,), dtype=torch.int32, device="cuda"),
        )

        def pack_once() -> None:
            deep_ep._C.host_staging_pack_fp8_dispatch_out(
                x_fp8, sf, topk_idx, topk_weights,
                outputs[0], outputs[1], outputs[2], outputs[3], outputs[4], outputs[5],
                rank_idx, args.num_ranks, args.num_scaleup_ranks, args.num_experts, args.num_sms)

        def packed_byte_views(actual_count: int) -> List[torch.Tensor]:
            return make_packed_byte_views(
                outputs, actual_count, args.hidden, sf.size(1), args.num_topk, topk_idx.element_size())
    elif args.payload_mode == "combine_bf16":
        x_bf16 = torch.randn((args.num_tokens, args.hidden), dtype=torch.bfloat16, device="cuda")
        outputs = (
            torch.empty_like(x_bf16),
            torch.empty((args.num_tokens,), dtype=torch.int32, device="cuda"),
            torch.empty((1,), dtype=torch.int32, device="cuda"),
        )

        def pack_once() -> None:
            deep_ep._C.host_staging_pack_bf16_combine_out(
                x_bf16, topk_idx, outputs[0], outputs[1], outputs[2],
                rank_idx, args.num_ranks, args.num_scaleup_ranks, args.num_experts, args.num_sms)

        def packed_byte_views(actual_count: int) -> List[torch.Tensor]:
            return make_bf16_combine_byte_views(outputs, actual_count, args.hidden)
    else:
        raise ValueError(f"Unsupported payload mode: {args.payload_mode}")

    for _ in range(args.warmups):
        pack_once()
    torch.cuda.synchronize()
    pack_start = time.perf_counter()
    pack_once()
    torch.cuda.synchronize()
    pack_s = time.perf_counter() - pack_start

    actual_count = int(outputs[-1].cpu().item())
    packed_views = packed_byte_views(actual_count)
    host_send = [
        torch.empty((view.numel(),), dtype=torch.uint8, pin_memory=True, device="cpu")
        for view in packed_views
    ]
    copy_streams = [torch.cuda.Stream(device=device) for _ in range(args.copy_streams)]

    torch.cuda.synchronize()
    d2h_start = time.perf_counter()
    copy_many(packed_views, host_send, copy_streams)
    torch.cuda.synchronize()
    d2h_s = time.perf_counter() - d2h_start

    lengths = [int(t.numel()) for t in host_send]
    send_header = {
        "node_rank": args.node_rank,
        "rank_idx": rank_idx,
        "peer_rank_idx": peer_rank_idx,
        "remote_tokens": actual_count,
        "lengths": lengths,
    }
    port = args.base_port + local_rank

    if args.node_rank == 0:
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.listen_host, port))
        server.listen(1)
        sock, _addr = server.accept()
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        server.close()
        transfer_start = time.perf_counter()
        recv_header = recv_json(sock)
        host_recv = [
            torch.empty((int(size),), dtype=torch.uint8, pin_memory=True, device="cpu")
            for size in recv_header["lengths"]
        ]
        send_json(sock, send_header)
    else:
        sock = connect_with_retry(args.peer_host, port, args.connect_timeout)
        send_json(sock, send_header)
        recv_header = recv_json(sock)
        host_recv = [
            torch.empty((int(size),), dtype=torch.uint8, pin_memory=True, device="cpu")
            for size in recv_header["lengths"]
        ]

    send_bytes = sum(lengths)
    recv_bytes = sum(int(size) for size in recv_header["lengths"])
    gpu_recv = [torch.empty_like(t, device="cuda") for t in host_recv]

    def timed_iteration() -> Dict[str, float]:
        torch.cuda.synchronize()
        pack_start = time.perf_counter()
        pack_once()
        torch.cuda.synchronize()
        pack_s = time.perf_counter() - pack_start

        d2h_start = time.perf_counter()
        copy_many(packed_views, host_send, copy_streams)
        torch.cuda.synchronize()
        d2h_s = time.perf_counter() - d2h_start

        transfer_start = time.perf_counter()
        exchange_tensors(sock, args.node_rank, host_send, host_recv)
        transfer_s = time.perf_counter() - transfer_start

        h2d_start = time.perf_counter()
        copy_many(host_recv, gpu_recv, copy_streams)
        torch.cuda.synchronize()
        h2d_s = time.perf_counter() - h2d_start
        return {
            "pack_s": pack_s,
            "d2h_s": d2h_s,
            "transfer_s": transfer_s,
            "h2d_s": h2d_s,
            "ep_s": pack_s + d2h_s + transfer_s + h2d_s,
        }

    for _ in range(args.socket_warmups):
        timed_iteration()
    samples = [timed_iteration() for _ in range(args.iters)]

    final_digest = {"digests": [tensor_digest(t) for t in host_send]}
    if args.node_rank == 0:
        recv_digest = recv_json(sock)
        send_json(sock, final_digest)
    else:
        send_json(sock, final_digest)
        recv_digest = recv_json(sock)
    sock.close()

    expected_digests = recv_digest["digests"]
    actual_digests = [tensor_digest(t) for t in host_recv]
    if actual_digests != expected_digests:
        raise RuntimeError(f"Digest mismatch on rank {rank_idx}: {actual_digests=} {expected_digests=}")

    if args.check_h2d:
        for host_tensor, gpu_tensor in zip(host_recv, gpu_recv):
            if host_tensor.numel() == 0:
                continue
            if not torch.equal(gpu_tensor.cpu(), host_tensor):
                raise RuntimeError(f"H2D mismatch on rank {rank_idx}")

    pack_stats = summarize_us([sample["pack_s"] for sample in samples])
    d2h_stats = summarize_us([sample["d2h_s"] for sample in samples])
    transfer_stats = summarize_us([sample["transfer_s"] for sample in samples])
    h2d_stats = summarize_us([sample["h2d_s"] for sample in samples])
    ep_stats = summarize_us([sample["ep_s"] for sample in samples])
    rdma_full_duplex_s = max(send_bytes, recv_bytes) / 1e9 / args.rdma_gbs_per_rank
    rdma_sequential_s = (send_bytes + recv_bytes) / 1e9 / args.rdma_gbs_per_rank
    rdma_est_ep_s = (
        statistics.median([sample["pack_s"] for sample in samples]) +
        statistics.median([sample["d2h_s"] for sample in samples]) +
        rdma_full_duplex_s +
        statistics.median([sample["h2d_s"] for sample in samples])
    )
    row = {
        "node_rank": args.node_rank,
        "rank_idx": rank_idx,
        "device": device,
        "peer_rank_idx": peer_rank_idx,
        "payload_mode": args.payload_mode,
        "send_remote_tokens": actual_count,
        "recv_remote_tokens": int(recv_header["remote_tokens"]),
        "send_bytes": send_bytes,
        "recv_bytes": recv_bytes,
        "iters": args.iters,
        "pack_median_us": pack_stats["median_us"],
        "pack_p90_us": pack_stats["p90_us"],
        "d2h_median_us": d2h_stats["median_us"],
        "d2h_p90_us": d2h_stats["p90_us"],
        "socket_transfer_median_us": transfer_stats["median_us"],
        "socket_transfer_p90_us": transfer_stats["p90_us"],
        "h2d_median_us": h2d_stats["median_us"],
        "h2d_p90_us": h2d_stats["p90_us"],
        "ep_socket_median_us": ep_stats["median_us"],
        "ep_socket_p90_us": ep_stats["p90_us"],
        "d2h_gbs": send_bytes / 1e9 / statistics.median([sample["d2h_s"] for sample in samples]),
        "socket_transfer_gbs": (
            (send_bytes + recv_bytes) / 1e9 / statistics.median([sample["transfer_s"] for sample in samples])
        ),
        "h2d_gbs": recv_bytes / 1e9 / statistics.median([sample["h2d_s"] for sample in samples]),
        "rdma_gbs_per_rank": args.rdma_gbs_per_rank,
        "rdma_full_duplex_est_us": rdma_full_duplex_s * 1e6,
        "rdma_sequential_est_us": rdma_sequential_s * 1e6,
        "ep_rdma_budget_est_us": rdma_est_ep_s * 1e6,
        "copy_streams": args.copy_streams,
        "num_sms": args.num_sms,
        "hidden": args.hidden,
        "num_topk": args.num_topk,
        "num_experts": args.num_experts,
    }
    # Preserve the original single-sample field names for quick grepping.
    row.update({
        "pack_us": row["pack_median_us"],
        "d2h_us": row["d2h_median_us"],
        "transfer_us": row["socket_transfer_median_us"],
        "h2d_us": row["h2d_median_us"],
    })
    queue.put(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-node explicit host-staging payload smoke for L4")
    parser.add_argument("--devices", type=str, default="0,1,2,3")
    parser.add_argument("--node-rank", type=int, required=True, choices=[0, 1])
    parser.add_argument("--listen-host", type=str, default="0.0.0.0")
    parser.add_argument("--peer-host", type=str, required=True)
    parser.add_argument("--base-port", type=int, default=28680)
    parser.add_argument("--connect-timeout", type=float, default=60.0)
    parser.add_argument("--num-ranks", type=int, default=8)
    parser.add_argument("--num-scaleup-ranks", type=int, default=4)
    parser.add_argument("--num-tokens", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=7168)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--num-sms", type=int, default=10)
    parser.add_argument("--copy-streams", type=int, default=2)
    parser.add_argument("--payload-mode", type=str, default="dispatch_fp8", choices=["dispatch_fp8", "combine_bf16"])
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--socket-warmups", type=int, default=1)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--rdma-gbs-per-rank", type=float, default=24.75)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--check-h2d", action="store_true")
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
