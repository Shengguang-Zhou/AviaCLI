"""Hardware/network-aware upload concurrency autotuning.

Picks folder-upload concurrency the way GPU batch size is picked from VRAM:
measure the host class (local/LAN/WAN) and round-trip time, then size the
connection pool from bandwidth-delay product, Little's Law, and CPU count so
users do not hand-tune ``--concurrency`` on every run.

The RTT probe uses the same explicit requests proxy snapshot as the PUT path.
"""

from __future__ import annotations

import math
import time
from urllib.parse import urlsplit

from avia_cli.core.uploads.transfer import UploadTransportRoute

__all__ = (
    "compute_cpu_upload_params",
    "compute_transport_concurrency",
    "compute_upload_params",
    "detect_storage_kind",
    "probe_rtt_seconds",
)

# Per-connection throughput ceiling (MB/s). The single load-bearing constant:
# every other concurrency driver is derived from it.
_PER_CONN_MBPS = {"local": 150.0, "lan": 80.0, "wan": 50.0}
# Aggregate throughput we aim to saturate (MB/s).
_TARGET_MBPS = {"local": 2000.0, "lan": 1000.0, "wan": 375.0}
_LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def detect_storage_kind(host: str) -> str:
    """Classify an upload host as ``local``, ``lan`` or ``wan``."""
    normalized = (host or "").strip().lower()
    if normalized in _LOCAL_HOSTS or "minio" in normalized:
        return "local"
    octets = normalized.split(".")
    if len(octets) == 4 and all(part.isdigit() for part in octets):
        a, b, *_ = (int(part) for part in octets)
        if a == 10:
            return "lan"
        if a == 192 and b == 168:
            return "lan"
        if a == 172 and 16 <= b <= 31:
            return "lan"
    return "wan"


def probe_rtt_seconds(route: UploadTransportRoute, storage_kind: str) -> float:
    """Measure three fresh HEAD round trips over the exact PUT transport route."""
    if storage_kind not in _PER_CONN_MBPS:
        raise ValueError(f"unsupported storage kind: {storage_kind}")
    parsed = urlsplit(route.upload_url)
    host = parsed.hostname
    if host is None:
        raise RuntimeError("signed upload URL must expose an absolute storage host")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    import requests

    best: float | None = None
    last_error: BaseException | None = None
    for _ in range(3):
        start = time.perf_counter()
        try:
            with requests.Session() as session:
                session.trust_env = False
                with session.head(
                    route.upload_url,
                    timeout=(2.0, 2.0),
                    allow_redirects=False,
                    stream=True,
                    proxies=route.request_proxies(),
                ) as response:
                    status = int(response.status_code)
                    if status == 407:
                        raise RuntimeError("RTT probe proxy authentication failed: status=407")
                    if 300 <= status < 400:
                        raise RuntimeError(f"RTT probe rejected storage redirect: status={status}")
                    if status in {408, 429} or status >= 500:
                        last_error = RuntimeError(
                            f"storage route returned retryable status={status}"
                        )
                        continue
                    elapsed = time.perf_counter() - start
        except requests.exceptions.RequestException as exc:
            last_error = exc
            continue
        if best is None or elapsed < best:
            best = elapsed
    if best is None:
        if last_error is None:
            raise RuntimeError(f"RTT probe failed for {host}:{port}: no successful attempt")
        if isinstance(last_error, RuntimeError):
            raise RuntimeError(f"RTT probe failed for {host}:{port}: {last_error}") from last_error
        raise RuntimeError(
            f"RTT probe failed for {host}:{port}: {type(last_error).__name__}"
        ) from last_error
    return best


def compute_upload_params(
    *,
    cores: int,
    storage_kind: str,
    probe_rtt_s: float,
    avg_file_bytes: int = 1_300_000,
) -> dict[str, int]:
    """Derive folder-upload concurrency/batch params from host class + RTT."""
    if storage_kind not in _PER_CONN_MBPS:
        raise ValueError(f"unsupported storage kind: {storage_kind}")
    if cores <= 0 or probe_rtt_s < 0 or avg_file_bytes <= 0:
        raise ValueError("cores and avg_file_bytes must be positive and RTT must be non-negative")
    return {
        **compute_cpu_upload_params(cores=cores),
        "concurrency": compute_transport_concurrency(
            cores=cores,
            storage_kind=storage_kind,
            probe_rtt_s=probe_rtt_s,
            avg_file_bytes=avg_file_bytes,
        ),
    }


def compute_cpu_upload_params(*, cores: int) -> dict[str, int]:
    if cores <= 0:
        raise ValueError("cores must be positive")
    return {
        "batch_complete_concurrency": _clamp(cores // 4, 2, 16),
        "hash_workers": _clamp(cores, 4, 16),
        "batch_size": 512,
        "stream_flush_size": 512,
    }


def compute_transport_concurrency(
    *,
    cores: int,
    storage_kind: str,
    probe_rtt_s: float,
    avg_file_bytes: int = 1_300_000,
) -> int:
    if storage_kind not in _PER_CONN_MBPS:
        raise ValueError(f"unsupported storage kind: {storage_kind}")
    if cores <= 0 or probe_rtt_s < 0 or avg_file_bytes <= 0:
        raise ValueError("cores and avg_file_bytes must be positive and RTT must be non-negative")
    per_conn_mbps = _PER_CONN_MBPS[storage_kind]
    target_mbps = _TARGET_MBPS[storage_kind]

    # AWS rule of thumb: one connection per ~per_conn_mbps of target throughput.
    conc_bw = math.ceil(target_mbps / per_conn_mbps)

    # Little's Law: enough in-flight requests to hide RTT behind transfer time.
    transfer_s = max(1e-4, avg_file_bytes / (per_conn_mbps * 1e6))
    conc_lat = math.ceil((probe_rtt_s + transfer_s) / transfer_s)

    # Multi-core hosts should never upload serially.
    cpu_floor = 2 * cores

    return _clamp(
        max(conc_bw, conc_lat, cpu_floor),
        8,
        min(256, 16 * cores),
    )
