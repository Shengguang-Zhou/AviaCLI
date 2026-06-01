from __future__ import annotations

import argparse
import http.client
import json
import os
import random
import socket
import sys
import time
from pathlib import Path
from typing import Any
from urllib import error as urlerror, parse, request

from avia_cli.core.errors import _AviaHTTPError, _UploadHTTPError
from avia_cli.core.auth.tokens import refresh_after_auth_error
from avia_cli.core.uploads.manifest import _image_size_file, _is_image_path, _sha256_file
from avia_cli.core.uploads.state import _ensure_sha256_batch as _support_ensure_sha256_batch
from avia_cli.core.uploads.state import _should_bypass_proxy_for_upload
from avia_cli.core.uploads.timing import put_file_with_retries as _retry_put_file
from avia_cli.core.uploads.transfer import put_file_curl as _transfer_put_file_curl
from avia_cli.core.uploads.transfer import put_file_requests as _transfer_put_file_requests

_SUCCESS_STATUSES = {"succeeded", "success", "completed", "complete", "ready", "done"}
_FAILED_STATUSES = {"failed", "error", "cancelled", "canceled"}
_UPLOAD_CHUNK_SIZE = 4 * 1024 * 1024
_DEFAULT_UPLOAD_READ_TIMEOUT = 45.0
_IMPORT_POLL_FAST_DELAYS_SEC = (0.25, 0.5, 1.0, 2.0, 4.0)

def _project_path(api: str, project_id: str, suffix: str) -> str:
    return (
        f"{api.rstrip('/')}/projects/{parse.quote(str(project_id), safe='')}/"
        f"{suffix.lstrip('/')}"
    )

def _ensure_sha256_batch(
    *,
    source_root: Path,
    files: list[dict[str, object]],
    hash_workers: int,
) -> list[dict[str, object]]:
    return _support_ensure_sha256_batch(
        source_root=source_root,
        files=files,
        hash_workers=hash_workers,
        sha256_file=_sha256_file,
        image_size_file=_image_size_file,
        is_image_path=_is_image_path,
    )


def _request_json(
    *,
    method: str,
    url: str,
    token: str,
    payload: dict[str, Any] | None = None,
    timeout: int | float = 60,
) -> dict[str, Any]:
    data = (
        json.dumps(payload, ensure_ascii=False).encode("utf-8")
        if payload is not None
        else None
    )
    for auth_attempt in range(2):
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        req = request.Request(url, data=data, headers=headers, method=method.upper())
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
            break
        except urlerror.HTTPError as exc:
            raw_body = exc.read()
            detail = raw_body.decode("utf-8", "replace") if raw_body else ""
            error = _AviaHTTPError(
                method=method,
                url=url,
                status=int(exc.code),
                reason=str(exc.reason),
                detail=detail[:4000],
            )
            if auth_attempt == 0 and refresh_after_auth_error(
                token, error, label=method
            ):
                continue
            raise error from exc
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def _request_json_with_retries(
    *,
    method: str,
    url: str,
    token: str,
    payload: dict[str, Any] | None = None,
    timeout: int | float = 60,
    retries: int = 3,
    label: str = "request",
) -> dict[str, Any]:
    attempts = max(1, int(retries or 1))
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return _request_json(
                method=method,
                url=url,
                token=token,
                payload=payload,
                timeout=timeout,
            )
        except Exception as exc:
            last_error = exc
            if refresh_after_auth_error(token, exc, label=label):
                continue
            if attempt + 1 >= attempts or not _is_transient_request_error(exc):
                raise
            delay = min(15.0, 0.8 * (2**attempt)) + random.uniform(0.0, 0.25)
            print(
                f"{label} failed transiently; retrying {attempt + 2}/{attempts} in {delay:.1f}s",
                file=sys.stderr,
            )
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def _post_json(
    *, api: str, token: str, project_id: str, payload: dict[str, object]
) -> dict:
    return _request_json(
        method="POST",
        url=_project_path(str(api), project_id, "imports/source"),
        token=token,
        payload=dict(payload),
        timeout=60,
    )


def _create_dataset_session(
    *,
    api: str,
    token: str,
    project_id: str,
    manifest: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, Any]:
    classes = list(args.class_name or []) or [
        str(item).strip()
        for item in list(manifest.get("classes") or [])
        if str(item).strip()
    ]
    payload = {
        "format": str(args.format),
        "root_name": Path(str(manifest["source"])).name,
        "task_key": str(args.task_key),
        "classes": classes,
        "file_count": int(manifest["file_count"]),
        "total_bytes": int(manifest["total_bytes"]),
        "auto_crop_embedding": bool(getattr(args, "auto_crop_embedding", True)),
    }
    return _request_json_with_retries(
        method="POST",
        url=_project_path(api, project_id, "imports/dataset-session"),
        token=token,
        payload=payload,
        timeout=60,
        retries=3,
        label="dataset-session",
    )


def _batch_upload_urls(
    *,
    api: str,
    token: str,
    project_id: str,
    import_id: str,
    files: list[dict[str, object]],
    timeout: int | float = 60,
    retries: int = 3,
) -> dict[str, Any]:
    payload = {"files": files}
    return _request_json_with_retries(
        method="POST",
        url=_project_path(
            api,
            project_id,
            f"imports/{parse.quote(import_id, safe='')}/files:batch-upload-urls",
        ),
        token=token,
        payload=payload,
        timeout=timeout,
        retries=retries,
        label="batch-upload-urls",
    )


def _complete_dataset_file_batch(
    *,
    api: str,
    token: str,
    project_id: str,
    import_id: str,
    files: list[dict[str, object]],
    timeout: int | float = 900,
    retries: int = 4,
) -> dict[str, Any]:
    url = _project_path(
        api,
        project_id,
        f"imports/{parse.quote(import_id, safe='')}/files:batch-complete",
    )
    attempts = max(1, int(retries or 1))
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return _request_json(
                method="POST",
                url=url,
                token=token,
                payload={"files": list(files)},
                timeout=timeout,
            )
        except Exception as exc:
            last_error = exc
            if refresh_after_auth_error(token, exc, label="batch-complete"):
                continue
            if attempt + 1 >= attempts or not _is_transient_request_error(exc):
                raise
            delay = min(30.0, 2.0 * (attempt + 1))
            print(
                "batch-complete timed out or failed transiently; "
                f"retrying {attempt + 2}/{attempts} in {delay:.1f}s",
                file=sys.stderr,
            )
            time.sleep(delay)
    assert last_error is not None
    raise last_error


def _is_transient_request_error(exc: Exception) -> bool:
    if isinstance(exc, _AviaHTTPError):
        return int(exc.status) in {408, 429, 500, 502, 503, 504}
    if isinstance(
        exc, (TimeoutError, socket.timeout, ConnectionError, http.client.HTTPException)
    ):
        return True
    if isinstance(exc, urlerror.HTTPError):
        return int(getattr(exc, "code", 0) or 0) in {408, 429, 500, 502, 503, 504}
    if isinstance(exc, urlerror.URLError):
        reason = getattr(exc, "reason", None)
        return isinstance(
            reason,
            (TimeoutError, socket.timeout, ConnectionError, http.client.HTTPException),
        )
    return False


def _put_file(
    *,
    upload_url: str,
    path: Path,
    headers: dict[str, object],
    connect_timeout: float = 15.0,
    read_timeout: float = _DEFAULT_UPLOAD_READ_TIMEOUT,
) -> None:
    clean_headers = {str(key): str(value) for key, value in dict(headers or {}).items()}
    clean_headers.setdefault("Content-Length", str(path.stat().st_size))
    parsed = parse.urlsplit(upload_url)
    if parsed.scheme not in {"http", "https"}:
        raise RuntimeError(f"unsupported upload URL scheme: {parsed.scheme}")
    signed_host = clean_headers.pop("Host", None)
    bypass_proxy = _should_bypass_proxy_for_upload(parsed)
    if _proxy_configured_for(parsed) and not bypass_proxy:
        try:
            _put_file_requests(
                upload_url=upload_url,
                path=path,
                headers=clean_headers,
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
            )
        except ImportError:
            _put_file_curl(
                upload_url=upload_url,
                path=path,
                headers=clean_headers,
                cause=RuntimeError("requests is not installed for proxy upload"),
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
            )
        return
    target = parsed.path or "/"
    if parsed.query:
        target = f"{target}?{parsed.query}"
    connection_cls = (
        http.client.HTTPSConnection
        if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
    conn = connection_cls(
        parsed.netloc, timeout=max(float(connect_timeout), float(read_timeout))
    )
    with path.open("rb") as fh:
        try:
            conn.putrequest("PUT", target, skip_host=signed_host is not None)
            if signed_host is not None:
                conn.putheader("Host", signed_host)
            for key, value in clean_headers.items():
                conn.putheader(key, value)
            conn.endheaders()
            for chunk in iter(lambda: fh.read(_UPLOAD_CHUNK_SIZE), b""):
                conn.send(chunk)
            resp = conn.getresponse()
            body = resp.read()
            if resp.status >= 400:
                detail = body.decode("utf-8", "replace")[:500]
                raise _UploadHTTPError(
                    status=int(resp.status),
                    reason=str(resp.reason),
                    detail=detail,
                )
        finally:
            conn.close()


def _is_retryable_upload_error(exc: BaseException) -> bool:
    if isinstance(exc, _UploadHTTPError):
        return exc.status in {408, 429, 500, 502, 503, 504}
    return True


def _put_file_with_retries(
    *,
    upload_url: str,
    path: Path,
    headers: dict[str, object],
    retries: int,
    base_delay_sec: float,
    connect_timeout: float = 15.0,
    read_timeout: float = _DEFAULT_UPLOAD_READ_TIMEOUT,
) -> None:
    _retry_put_file(
        put_file=_put_file,
        is_retryable=_is_retryable_upload_error,
        upload_url=upload_url,
        path=path,
        headers=headers,
        retries=retries,
        base_delay_sec=base_delay_sec,
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
    )


def _proxy_configured_for(parsed: parse.SplitResult) -> bool:
    lowercase_all_proxy = os.environ.get("all_proxy")
    proxy = (
        os.environ.get(f"{parsed.scheme}_proxy")
        or os.environ.get(f"{parsed.scheme.upper()}_PROXY")
        or os.environ.get("ALL_PROXY")
        or lowercase_all_proxy
    )
    if not proxy:
        return False
    no_proxy = os.environ.get("no_proxy") or os.environ.get("NO_PROXY") or ""
    host = str(parsed.hostname or "").lower()
    for raw in no_proxy.split(","):
        entry = raw.strip().lower()
        if not entry:
            continue
        if entry in {"*", host} or (entry.startswith(".") and host.endswith(entry)):
            return False
    return True


def _put_file_requests(
    *,
    upload_url: str,
    path: Path,
    headers: dict[str, object],
    connect_timeout: float = 15.0,
    read_timeout: float = _DEFAULT_UPLOAD_READ_TIMEOUT,
) -> None:
    _transfer_put_file_requests(
        upload_url=upload_url,
        path=path,
        headers=headers,
        upload_error=_UploadHTTPError,
        curl_callback=_put_file_curl,
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
    )


def _put_file_curl(
    *,
    upload_url: str,
    path: Path,
    headers: dict[str, object],
    cause: BaseException,
    connect_timeout: float = 15.0,
    read_timeout: float = _DEFAULT_UPLOAD_READ_TIMEOUT,
) -> None:
    _transfer_put_file_curl(
        upload_url=upload_url,
        path=path,
        headers=headers,
        cause=cause,
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
    )


def _complete_import(
    *, api: str, token: str, project_id: str, import_id: str
) -> dict[str, Any]:
    return _request_json_with_retries(
        method="POST",
        url=_project_path(
            api, project_id, f"imports/{parse.quote(import_id, safe='')}/complete"
        ),
        token=token,
        payload={},
        timeout=60,
        retries=3,
        label="complete-import",
    )


def _poll_import(
    *,
    api: str,
    token: str,
    project_id: str,
    import_id: str,
    timeout_sec: int,
    interval_sec: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + max(1, int(timeout_sec))
    last: dict[str, Any] = {}
    poll_attempt = 0
    while True:
        last = _request_json_with_retries(
            method="GET",
            url=_project_path(
                api, project_id, f"ingestion-jobs/{parse.quote(import_id, safe='')}"
            ),
            token=token,
            timeout=60,
            retries=2,
            label="poll-import",
        )
        status = str(last.get("status") or "").strip().lower()
        if status in _SUCCESS_STATUSES or status in _FAILED_STATUSES:
            return last
        if time.monotonic() >= deadline:
            raise SystemExit(
                f"timed out waiting for import {import_id}; last status={status or 'unknown'}"
            )
        configured_interval = max(0.1, float(interval_sec or 1.0))
        fast_delay = _IMPORT_POLL_FAST_DELAYS_SEC[
            min(poll_attempt, len(_IMPORT_POLL_FAST_DELAYS_SEC) - 1)
        ]
        poll_attempt += 1
        time.sleep(min(configured_interval, fast_delay))
