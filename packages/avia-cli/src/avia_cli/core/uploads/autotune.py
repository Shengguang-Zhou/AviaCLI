"""Hardware/network-aware upload concurrency autotuning.

Picks folder-upload concurrency the way GPU batch size is picked from VRAM:
measure the host class (local/LAN/WAN) and round-trip time, then size the
connection pool from bandwidth-delay product, Little's Law, and CPU count so
users do not hand-tune ``--concurrency`` on every run.

Pure stdlib, no third-party dependencies.
"""

from __future__ import annotations

import math
import socket
import time

__all__ = (
    "compute_upload_params",
    "detect_storage_kind",
    "probe_rtt_seconds",
)

# Per-connection throughput ceiling (MB/s). The single load-bearing constant:
# every other concurrency driver is derived from it.
_PER_CONN_MBPS = {"local": 150.0, "lan": 80.0, "wan": 50.0}
# Aggregate throughput we aim to saturate (MB/s).
_TARGET_MBPS = {"local": 2000.0, "lan": 1000.0, "wan": 375.0}
# RTT fallbacks when the TCP probe cannot connect (seconds).
_FALLBACK_RTT = {"local": 0.0005, "lan": 0.002, "wan": 0.03}

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


def probe_rtt_seconds(host: str, port: int, storage_kind: str) -> float:
    """Measure TCP connect RTT (min of 3 tries); fall back on any failure."""
    fallback = _FALLBACK_RTT.get(storage_kind, _FALLBACK_RTT["wan"])
    best: float | None = None
    for _ in range(3):
        start = time.perf_counter()
        try:
            with socket.create_connection((host, port), timeout=2.0):
                pass
        except OSError:
            return fallback
        elapsed = time.perf_counter() - start
        if best is None or elapsed < best:
            best = elapsed
    return best if best is not None else fallback


def compute_upload_params(
    *,
    cores: int,
    storage_kind: str,
    probe_rtt_s: float,
    avg_file_bytes: int = 1_300_000,
) -> dict[str, int]:
    """Derive folder-upload concurrency/batch params from host class + RTT."""
    per_conn_mbps = _PER_CONN_MBPS.get(storage_kind, _PER_CONN_MBPS["lan"])
    target_mbps = _TARGET_MBPS.get(storage_kind, _TARGET_MBPS["lan"])

    # AWS rule of thumb: one connection per ~per_conn_mbps of target throughput.
    conc_bw = math.ceil(target_mbps / per_conn_mbps)

    # Little's Law: enough in-flight requests to hide RTT behind transfer time.
    transfer_s = max(1e-4, avg_file_bytes / (per_conn_mbps * 1e6))
    conc_lat = math.ceil((probe_rtt_s + transfer_s) / transfer_s)

    # Multi-core hosts should never upload serially.
    cpu_floor = 2 * cores

    concurrency = _clamp(
        max(conc_bw, conc_lat, cpu_floor),
        8,
        min(256, 16 * cores),
    )
    return {
        "concurrency": concurrency,
        "batch_complete_concurrency": _clamp(cores // 4, 2, 16),
        "hash_workers": _clamp(cores, 4, 16),
        "batch_size": 512,
        "stream_flush_size": 512,
    }
