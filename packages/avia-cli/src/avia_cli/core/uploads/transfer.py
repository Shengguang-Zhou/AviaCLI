from __future__ import annotations

import http.client
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any
from urllib import parse


def put_file_requests(
    *,
    upload_url: str,
    path: Path,
    headers: dict[str, object],
    upload_error: type[RuntimeError],
    curl_callback: Any,
    connect_timeout: float,
    read_timeout: float,
) -> None:
    import requests

    request_headers = {
        str(key): str(value) for key, value in dict(headers or {}).items()
    }

    def put_once(*, trust_env: bool):
        with requests.Session() as session:
            session.trust_env = trust_env
            with path.open("rb") as fh:
                return session.put(
                    upload_url,
                    data=fh,
                    headers=request_headers,
                    timeout=(float(connect_timeout), float(read_timeout)),
                )

    try:
        resp = put_once(trust_env=True)
    except requests.exceptions.RequestException:
        try:
            resp = put_once(trust_env=False)
        except requests.exceptions.RequestException as direct_error:
            curl_callback(
                upload_url=upload_url,
                path=path,
                headers=request_headers,
                cause=direct_error,
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
            )
            return
    if resp.status_code >= 400:
        raise upload_error(
            status=int(resp.status_code),
            reason=str(resp.reason or ""),
            detail=resp.text[:500],
        )


def put_file_curl(
    *,
    upload_url: str,
    path: Path,
    headers: dict[str, object],
    cause: BaseException,
    connect_timeout: float,
    read_timeout: float,
) -> None:
    curl = shutil.which("curl")
    if not curl:
        raise cause
    config_lines = [f"url = {_curl_config_quote(upload_url)}"]
    for key, value in dict(headers or {}).items():
        config_lines.append(f"header = {_curl_config_quote(f'{key}: {value}')}")
    proc = subprocess.run(
        [
            curl,
            "-fsS",
            "-X",
            "PUT",
            "--connect-timeout",
            str(max(1.0, float(connect_timeout))),
            "--max-time",
            str(max(1.0, float(read_timeout))),
            "--config",
            "-",
            "--data-binary",
            "@" + str(path),
        ],
        input="\n".join(config_lines) + "\n",
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip()[:500]
        raise RuntimeError(
            f"upload failed via curl: exit {proc.returncode}: {detail}"
        ) from cause


def _curl_config_quote(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "") + '"'


def put_file_part(
    *,
    upload_url: str,
    path: Path,
    offset: int,
    size: int,
    headers: dict[str, object],
    upload_chunk_size: int,
) -> str:
    clean_headers = {str(key): str(value) for key, value in dict(headers or {}).items()}
    clean_headers["Content-Length"] = str(int(size))
    parsed = parse.urlsplit(upload_url)
    if parsed.scheme not in {"http", "https"}:
        raise RuntimeError(f"unsupported upload URL scheme: {parsed.scheme}")
    target = parsed.path or "/"
    if parsed.query:
        target = f"{target}?{parsed.query}"
    connection_cls = (
        http.client.HTTPSConnection
        if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
    conn = connection_cls(parsed.netloc, timeout=600)
    remaining = int(size)
    with path.open("rb") as fh:
        fh.seek(int(offset))
        try:
            conn.putrequest("PUT", target)
            for key, value in clean_headers.items():
                conn.putheader(key, value)
            conn.endheaders()
            while remaining > 0:
                chunk = fh.read(min(upload_chunk_size, remaining))
                if not chunk:
                    break
                conn.send(chunk)
                remaining -= len(chunk)
            if remaining != 0:
                raise RuntimeError(
                    f"short read while uploading part: remaining={remaining}"
                )
            resp = conn.getresponse()
            body = resp.read()
            if resp.status >= 400:
                detail = body.decode("utf-8", "replace")[:500]
                raise RuntimeError(
                    f"part upload failed: HTTP {resp.status} {resp.reason}: {detail}"
                )
            etag = str(resp.getheader("ETag") or "").strip().strip('"')
            if not etag:
                raise RuntimeError("part upload response did not include ETag")
            return etag
        finally:
            conn.close()


def put_file_part_with_retries(
    *,
    put_part_callback: Any,
    upload_url: str,
    path: Path,
    offset: int,
    size: int,
    headers: dict[str, object],
    retries: int,
    base_delay_sec: float,
) -> str:
    attempts = max(1, int(retries or 1))
    delay = max(0.1, float(base_delay_sec or 0.1))
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return put_part_callback(
                upload_url=upload_url,
                path=path,
                offset=offset,
                size=size,
                headers=headers,
            )
        except Exception as exc:
            last_error = exc
            if attempt >= attempts:
                break
            time.sleep(min(30.0, delay * (2 ** (attempt - 1))))
    assert last_error is not None
    raise last_error
