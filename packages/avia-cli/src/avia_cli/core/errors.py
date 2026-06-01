from __future__ import annotations

import json


class _UploadHTTPError(RuntimeError):
    def __init__(self, *, status: int, reason: str, detail: str) -> None:
        self.status = int(status)
        self.reason = str(reason)
        self.detail = str(detail)
        super().__init__(f"upload failed: HTTP {self.status} {self.reason}: {self.detail}")


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
    try:
        payload = json.loads(message) if message else {}
        error = dict(payload.get("error") or {})
        message = str(error.get("message") or message).strip()
        correlation_id = str(error.get("correlation_id") or "").strip()
    except Exception:
        pass
    parts = [f"Error: {message or exc.reason}", f"HTTP {exc.status} {exc.reason}"]
    if correlation_id:
        parts.append(f"correlation_id={correlation_id}")
    parts.append(exc.url)
    return " | ".join(parts)
