from __future__ import annotations

import sys
import threading
import time
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")
_MIB = 1024 * 1024


class UploadTimingRecorder:
    def __init__(self, *, first_file_target: int = 512) -> None:
        self._first_file_target = int(first_file_target)
        self._accepted_files = 0
        self._first_accepted: dict[str, Any] | None = None
        self._events: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def record(
        self,
        name: str,
        *,
        duration_sec: float,
        file_count: int = 0,
        byte_count: int = 0,
    ) -> None:
        duration = max(0.0, float(duration_sec))
        with self._lock:
            event = self._events.setdefault(
                str(name),
                {
                    "count": 0,
                    "duration_sec": 0.0,
                    "max_duration_sec": 0.0,
                    "file_count": 0,
                    "byte_count": 0,
                },
            )
            event["count"] = int(event["count"]) + 1
            event["duration_sec"] = float(event["duration_sec"]) + duration
            event["max_duration_sec"] = max(float(event["max_duration_sec"]), duration)
            event["file_count"] = int(event["file_count"]) + int(file_count)
            event["byte_count"] = int(event["byte_count"]) + int(byte_count)

    def time_call(
        self,
        name: str,
        fn: Callable[..., T],
        *,
        file_count: int = 0,
        byte_count: int = 0,
        **kwargs: Any,
    ) -> T:
        started = time.monotonic()
        try:
            return fn(**kwargs)
        finally:
            self.record(
                name,
                duration_sec=time.monotonic() - started,
                file_count=file_count,
                byte_count=byte_count,
            )

    def record_accepted_files(self, count: int, *, elapsed_sec: float) -> None:
        with self._lock:
            self._accepted_files += max(0, int(count))
            if self._first_accepted is not None:
                return
            if self._accepted_files >= self._first_file_target:
                self._first_accepted = {
                    "target_files": self._first_file_target,
                    "reached": True,
                    "elapsed_sec": round(max(0.0, float(elapsed_sec)), 3),
                    "accepted_files": self._accepted_files,
                }

    def summary(self) -> dict[str, Any]:
        with self._lock:
            result = {
                name: _summarize_event(event)
                for name, event in sorted(self._events.items())
            }
            if self._first_accepted is not None:
                result["first_512_accepted"] = dict(self._first_accepted)
            elif self._accepted_files > 0:
                result["first_512_accepted"] = {
                    "target_files": self._first_file_target,
                    "reached": False,
                    "accepted_files": self._accepted_files,
                }
            return result


def _summarize_event(event: dict[str, Any]) -> dict[str, Any]:
    count = int(event.get("count") or 0)
    duration = round(float(event.get("duration_sec") or 0.0), 6)
    byte_count = int(event.get("byte_count") or 0)
    return {
        "count": count,
        "duration_sec": duration,
        "avg_duration_sec": round(duration / count, 6) if count else None,
        "max_duration_sec": round(float(event.get("max_duration_sec") or 0.0), 6),
        "file_count": int(event.get("file_count") or 0),
        "byte_count": byte_count,
        "mib": round(byte_count / _MIB, 3),
        "throughput_mibps": (
            round((byte_count / _MIB) / duration, 3)
            if byte_count > 0 and duration > 0
            else None
        ),
    }


def put_file_with_retries(
    *,
    put_file: Callable[..., None],
    is_retryable: Callable[[BaseException], bool],
    upload_url: str,
    path: Any,
    headers: dict[str, object],
    retries: int,
    base_delay_sec: float,
    connect_timeout: float,
    read_timeout: float,
) -> None:
    attempts = max(1, int(retries or 1))
    delay = max(0.1, float(base_delay_sec or 0.1))
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            put_file(
                upload_url=upload_url,
                path=path,
                headers=headers,
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
            )
            return
        except Exception as exc:
            last_error = exc
            if attempt >= attempts or not is_retryable(exc):
                break
            retry_number = attempt + 1
            sleep_sec = min(30.0, delay * (2 ** (attempt - 1)))
            print(
                f"PUT failed transiently; retrying {retry_number}/{attempts} in {sleep_sec:.1f}s",
                file=sys.stderr,
            )
            time.sleep(sleep_sec)
    assert last_error is not None
    raise last_error
