from __future__ import annotations

import http.client
import json
import random
import socket
import sys
import time
from pathlib import Path
from typing import Any
from urllib import error as urlerror, parse, request

from avia_cli.core.errors import (
    _AviaHTTPError,
    _UploadHTTPError,
    _is_retryable_upload_error,
    decode_json_response,
)
from avia_cli.core.auth.tokens import refresh_after_auth_error
from avia_cli.core.http import no_redirect
from avia_cli.core.uploads.inventory import is_dataset_image_path
from avia_cli.core.uploads.media_types import require_canonical_media_type
from avia_cli.core.uploads.response_contracts import (
    IMPORT_TERMINAL_STATUSES,
    decode_batch_complete_response,
    decode_batch_upload_urls_response,
    decode_complete_import_response,
    decode_dataset_session_response,
    decode_import_job_response,
    validate_batch_upload_urls_request,
)
from avia_cli.core.uploads.state import _ensure_sha256_batch as _support_ensure_sha256_batch
from avia_cli.core.uploads.source_file import SourceIdentity, VerifiedSourceFile
from avia_cli.core.uploads.timing import put_file_with_retries as _retry_put_file
from avia_cli.core.uploads.transfer import (
    UploadTransportRoute,
    put_file_requests as _transfer_put_file_requests,
)

_DEFAULT_UPLOAD_READ_TIMEOUT = 45.0
_IMPORT_POLL_FAST_DELAYS_SEC = (0.25, 0.5, 1.0, 2.0, 4.0)


def _project_path(api: str, project_id: str, suffix: str) -> str:
    return (
        f"{api.rstrip('/')}/projects/{parse.quote(str(project_id), safe='')}/{suffix.lstrip('/')}"
    )


def _ensure_sha256_batch(
    *,
    source_root: Path,
    files: list[dict[str, object]],
    hash_workers: int,
    source_identities: dict[str, SourceIdentity],
) -> list[dict[str, object]]:
    return _support_ensure_sha256_batch(
        source_root=source_root,
        files=files,
        hash_workers=hash_workers,
        source_identities=source_identities,
        is_image_path=is_dataset_image_path,
    )


def _request_json(
    *,
    method: str,
    url: str,
    token: str,
    payload: dict[str, Any] | None = None,
    timeout: int | float = 60,
) -> dict[str, Any]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") if payload is not None else None
    for auth_attempt in range(2):
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        if payload is not None:
            headers["Content-Type"] = "application/json"
        req = request.Request(url, data=data, headers=headers, method=method.upper())
        try:
            with no_redirect.open_no_redirect(req, timeout=float(timeout)) as resp:
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
            if auth_attempt == 0 and refresh_after_auth_error(token, error, label=method):
                continue
            raise error from exc
    if not raw:
        return {}
    return decode_json_response(raw, url=url)


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
    attempts = int(retries)
    if attempts <= 0:
        raise ValueError("request retries must be greater than zero")
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


def _post_json(*, api: str, token: str, project_id: str, payload: dict[str, object]) -> dict:
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
    payload: dict[str, object],
) -> dict[str, Any]:
    response = _request_json_with_retries(
        method="POST",
        url=_project_path(api, project_id, "imports/dataset-session"),
        token=token,
        payload=dict(payload),
        timeout=60,
        retries=3,
        label="dataset-session",
    )
    return decode_dataset_session_response(response, project_id=project_id)


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
    validate_batch_upload_urls_request(files)
    payload = {"files": [dict(item) for item in files]}
    response = _request_json_with_retries(
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
    return decode_batch_upload_urls_response(
        response,
        project_id=project_id,
        import_id=import_id,
        requested_files=files,
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
    for item in files:
        relative_path = str(item.get("relative_path") or "")
        require_canonical_media_type(
            item.get("content_type"),
            label=f"batch-complete content_type for {relative_path}",
        )
    url = _project_path(
        api,
        project_id,
        f"imports/{parse.quote(import_id, safe='')}/files:batch-complete",
    )
    attempts = int(retries)
    if attempts <= 0:
        raise ValueError("batch-complete retries must be greater than zero")
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            response = _request_json(
                method="POST",
                url=url,
                token=token,
                payload={"files": list(files)},
                timeout=timeout,
            )
            return decode_batch_complete_response(
                response,
                project_id=project_id,
                import_id=import_id,
                requested_paths=[str(item.get("relative_path") or "") for item in files],
            )
        except Exception as exc:
            last_error = exc
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
    if isinstance(exc, (TimeoutError, socket.timeout, ConnectionError, http.client.HTTPException)):
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
    route: UploadTransportRoute,
    source: VerifiedSourceFile,
    headers: dict[str, object],
    connect_timeout: float = 15.0,
    read_timeout: float = _DEFAULT_UPLOAD_READ_TIMEOUT,
) -> str:
    return _transfer_put_file_requests(
        route=route,
        source=source,
        headers=headers,
        upload_error=_UploadHTTPError,
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
    )


def _put_file_with_retries(
    *,
    route: UploadTransportRoute,
    path: Path,
    expected_identity: SourceIdentity | dict[str, object],
    headers: dict[str, object],
    retries: int,
    base_delay_sec: float,
    connect_timeout: float = 15.0,
    read_timeout: float = _DEFAULT_UPLOAD_READ_TIMEOUT,
) -> str:
    return _retry_put_file(
        put_file=_put_file,
        is_retryable=_is_retryable_upload_error,
        route=route,
        path=path,
        expected_identity=expected_identity,
        headers=headers,
        retries=retries,
        base_delay_sec=base_delay_sec,
        connect_timeout=connect_timeout,
        read_timeout=read_timeout,
    )


def _complete_import(*, api: str, token: str, project_id: str, import_id: str) -> dict[str, Any]:
    response = _request_json_with_retries(
        method="POST",
        url=_project_path(api, project_id, f"imports/{parse.quote(import_id, safe='')}/complete"),
        token=token,
        payload={},
        timeout=60,
        retries=3,
        label="complete-import",
    )
    return decode_complete_import_response(response, project_id=project_id, import_id=import_id)


def _poll_import(
    *,
    api: str,
    token: str,
    project_id: str,
    import_id: str,
    timeout_sec: int,
    interval_sec: float,
) -> dict[str, Any]:
    timeout = int(timeout_sec)
    configured_interval = float(interval_sec)
    if timeout <= 0 or configured_interval <= 0:
        raise ValueError("poll timeout and interval must be greater than zero")
    deadline = time.monotonic() + timeout
    last: dict[str, Any] = {}
    poll_attempt = 0
    while True:
        response = _request_json_with_retries(
            method="GET",
            url=_project_path(api, project_id, f"ingestion-jobs/{parse.quote(import_id, safe='')}"),
            token=token,
            timeout=60,
            retries=2,
            label="poll-import",
        )
        last = decode_import_job_response(response, project_id=project_id, import_id=import_id)
        status = str(last["status"])
        if status in IMPORT_TERMINAL_STATUSES:
            return last
        if time.monotonic() >= deadline:
            raise SystemExit(f"timed out waiting for import {import_id}; last status={status}")
        fast_delay = _IMPORT_POLL_FAST_DELAYS_SEC[
            min(poll_attempt, len(_IMPORT_POLL_FAST_DELAYS_SEC) - 1)
        ]
        poll_attempt += 1
        time.sleep(min(configured_interval, fast_delay))
