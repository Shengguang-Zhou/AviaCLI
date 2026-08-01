from __future__ import annotations

import json
from typing import Any

from avia_cli.core.strict_json import strict_json_loads


class _UploadHTTPError(RuntimeError):
    def __init__(self, *, status: int, reason: str, detail: str) -> None:
        self.status = int(status)
        self.reason = str(reason)
        self.detail = str(detail)
        super().__init__(f"upload failed: HTTP {self.status} {self.reason}: {self.detail}")


class _UploadTransportError(RuntimeError):
    def __init__(self, operation: str) -> None:
        self.operation = str(operation)
        super().__init__(f"{self.operation} transport failed")


class ConcurrentUploadError(RuntimeError):
    def __init__(self, failures: list[tuple[str, list[str], BaseException]]) -> None:
        if not failures:
            raise ValueError("concurrent upload failures must not be empty")
        payload = {
            "code": "concurrent_upload_failed",
            "failure_count": len(failures),
            "failures": [
                {
                    "phase": phase,
                    "targets": targets,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
                for phase, targets, exc in failures
            ],
        }
        self.failures = failures
        super().__init__(json.dumps(payload, ensure_ascii=False, sort_keys=True))


def _is_retryable_upload_error(exc: BaseException) -> bool:
    if isinstance(exc, _UploadTransportError):
        return True
    if isinstance(exc, _UploadHTTPError):
        return exc.status in {408, 429, 500, 502, 503, 504}
    return False


class _AviaHTTPError(RuntimeError):
    def __init__(
        self,
        *,
        method: str,
        url: str,
        status: int,
        reason: str,
        detail: str,
    ) -> None:
        self.method = str(method).upper()
        self.url = str(url)
        self.status = int(status)
        self.reason = str(reason)
        self.detail = str(detail)
        super().__init__(
            f"{self.method} {self.url} failed: HTTP {self.status} {self.reason}: {self.detail}"
        )


def format_avia_http_error(exc: _AviaHTTPError) -> str:
    message = str(exc.detail or "").strip()
    correlation_id = ""
    payload: object = None
    if message:
        try:
            payload = strict_json_loads(message)
        except ValueError:
            payload = None
    if isinstance(payload, dict) and isinstance(payload.get("error"), dict):
        error = payload["error"]
        message = str(error.get("message") or message).strip()
        correlation_id = str(error.get("correlation_id") or "").strip()
    parts = [f"Error: {message or exc.reason}", f"HTTP {exc.status} {exc.reason}"]
    if correlation_id:
        parts.append(f"correlation_id={correlation_id}")
    parts.append(exc.url)
    return " | ".join(parts)


def decode_json_response(raw: bytes, *, url: str) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(
            json.dumps(
                {
                    "code": "invalid_json_response",
                    "message": "HTTP response is not valid UTF-8 JSON",
                    "url": url,
                    "byte_offset": exc.start,
                },
                ensure_ascii=False,
            )
        ) from exc
    try:
        payload = strict_json_loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            json.dumps(
                {
                    "code": "invalid_json_response",
                    "message": "HTTP response body is malformed JSON",
                    "url": url,
                    "line": exc.lineno,
                    "column": exc.colno,
                },
                ensure_ascii=False,
            )
        ) from exc
    except ValueError as exc:
        raise RuntimeError(
            json.dumps(
                {
                    "code": "invalid_json_response",
                    "message": "HTTP response violates the strict JSON contract",
                    "url": url,
                    "reason": str(exc),
                },
                ensure_ascii=False,
            )
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(
            json.dumps(
                {
                    "code": "invalid_json_response",
                    "message": "HTTP JSON response must be an object",
                    "url": url,
                },
                ensure_ascii=False,
            )
        )
    return payload
