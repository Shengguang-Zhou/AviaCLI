from __future__ import annotations

import concurrent.futures
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeAlias

PutFailure: TypeAlias = tuple[str, list[str], BaseException]


@dataclass(frozen=True, slots=True)
class PutSuccess:
    """One completed remote PUT and its post-success telemetry outcome."""

    relative_path: str
    signed: dict[str, Any]
    telemetry_error: BaseException | None = None


def settle_concurrent_puts(
    *,
    relative_paths: list[str],
    max_workers: int,
    upload_one: Callable[[str], PutSuccess],
    record_success: Callable[[str, dict[str, Any]], None],
) -> list[PutFailure]:
    """Wait for every submitted PUT and account for every successful side effect."""

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    futures: dict[concurrent.futures.Future[PutSuccess], str] = {}
    settled: set[concurrent.futures.Future[PutSuccess]] = set()
    failures: list[PutFailure] = []

    def cancel_not_started() -> None:
        for future in futures:
            future.cancel()

    def fail(phase: str, path: str, exc: BaseException) -> None:
        failures.append((phase, [path] if path else [], exc))
        cancel_not_started()

    def settle(future: concurrent.futures.Future[PutSuccess]) -> None:
        expected_path = futures[future]
        try:
            result = future.result()
        except concurrent.futures.CancelledError as exc:
            if not failures:
                fail("file-put", expected_path, exc)
            return
        except Exception as exc:
            fail("file-put", expected_path, exc)
            return
        if result.relative_path != expected_path:
            fail(
                "file-put-state",
                expected_path,
                RuntimeError("file PUT returned a different relative path"),
            )
            return
        try:
            record_success(result.relative_path, result.signed)
        except Exception as exc:
            fail("file-put-state", expected_path, exc)
        if result.telemetry_error is not None:
            fail("file-put-telemetry", expected_path, result.telemetry_error)

    try:
        for relative_path in relative_paths:
            try:
                future = executor.submit(upload_one, relative_path)
            except Exception as exc:
                fail("file-put-submit", relative_path, exc)
                break
            futures[future] = relative_path
        for future in concurrent.futures.as_completed(tuple(futures)):
            settled.add(future)
            settle(future)
    except Exception as exc:
        fail("file-put-settlement", "", exc)
    finally:
        try:
            executor.shutdown(wait=True, cancel_futures=True)
        except Exception as exc:
            failures.append(("file-put-shutdown", [], exc))

    for future in futures:
        if future not in settled:
            settle(future)
    return failures


__all__ = ["PutFailure", "PutSuccess", "settle_concurrent_puts"]
