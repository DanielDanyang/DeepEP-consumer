import argparse
import json
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, List, Optional


BW_RE = re.compile(
    r"^\s*(?P<bytes>\d+)\s+(?P<iters>\d+)\s+"
    r"(?P<peak>[0-9.]+)\s+(?P<average>[0-9.]+)\s+(?P<msg_rate>[0-9.]+)\s*$"
)


@dataclass
class RunSpec:
    device: str
    port: int


def perftest_command(binary: str, spec: RunSpec, args: argparse.Namespace, host: Optional[str] = None) -> List[str]:
    cmd = [
        binary,
        "-d", spec.device,
        "-F",
        "-q", str(args.qps),
        "-s", str(args.size),
        "-n", str(args.iters),
        "--report_gbits",
        "-p", str(spec.port),
    ]
    if host is not None:
        cmd.append(host)
    return cmd


def start_local_server(binary: str, spec: RunSpec, args: argparse.Namespace) -> subprocess.Popen:
    return subprocess.Popen(
        perftest_command(binary, spec, args),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def start_remote_client(binary: str, spec: RunSpec, args: argparse.Namespace) -> subprocess.Popen:
    remote_cmd = shlex.join(perftest_command(binary, spec, args, args.server_addr))
    return subprocess.Popen(
        ["ssh", args.remote, remote_cmd],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def parse_bandwidth(output: str) -> Dict[str, float]:
    for line in output.splitlines():
        match = BW_RE.match(line)
        if match:
            avg_gbits = float(match.group("average"))
            return {
                "bytes": int(match.group("bytes")),
                "iterations": int(match.group("iters")),
                "peak_gbits": float(match.group("peak")),
                "avg_gbits": avg_gbits,
                "avg_gbs": avg_gbits / 8.0,
                "msg_rate_mpps": float(match.group("msg_rate")),
            }
    raise RuntimeError(f"Could not parse perftest bandwidth output:\n{output}")


def collect_process(proc: subprocess.Popen, timeout: int) -> str:
    try:
        output, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        output, _ = proc.communicate(timeout=5)
        raise RuntimeError(f"Command timed out after {timeout}s:\n{output}")
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}:\n{output}")
    return output


def run_specs(specs: List[RunSpec], args: argparse.Namespace) -> List[Dict[str, object]]:
    binary = f"ib_{args.op}_bw"
    servers = [(spec, start_local_server(binary, spec, args)) for spec in specs]
    # Perftest servers need a short window to bind before clients connect.
    time.sleep(args.connect_delay)

    clients = [(spec, start_remote_client(binary, spec, args)) for spec in specs]
    rows = []
    client_outputs: Dict[str, str] = {}
    for spec, proc in clients:
        client_outputs[spec.device] = collect_process(proc, args.timeout)
    for spec, proc in servers:
        server_output = collect_process(proc, args.timeout)
        row = parse_bandwidth(client_outputs[spec.device])
        row.update({
            "op": args.op,
            "device": spec.device,
            "port": spec.port,
            "qps": args.qps,
            "size": args.size,
            "requested_iters": args.iters,
            "parallel": len(specs) > 1,
            "remote": args.remote,
            "server_addr": args.server_addr,
            "server_output_tail": "\n".join(server_output.splitlines()[-6:]),
        })
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark two-node host-memory IB bandwidth for L4 host staging")
    parser.add_argument("--remote", type=str, default="l41")
    parser.add_argument("--server-addr", type=str, default="10.10.55.1")
    parser.add_argument("--devices", type=str, nargs="+", default=["mlx5_0", "mlx5_1"])
    parser.add_argument("--op", type=str, choices=["write", "read"], default="write")
    parser.add_argument("--qps", type=int, default=4)
    parser.add_argument("--size", type=int, default=8 * 1024 * 1024)
    parser.add_argument("--iters", type=int, default=2000)
    parser.add_argument("--base-port", type=int, default=18540)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--connect-delay", type=float, default=1.0)
    parser.add_argument("--parallel", action="store_true")
    args = parser.parse_args()

    specs = [RunSpec(device=device, port=args.base_port + idx) for idx, device in enumerate(args.devices)]
    groups = [specs] if args.parallel else [[spec] for spec in specs]
    for group in groups:
        for row in run_specs(group, args):
            print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
