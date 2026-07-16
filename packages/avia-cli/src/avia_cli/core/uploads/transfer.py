from __future__ import annotations

import re
from urllib import parse

from avia_cli.core.errors import _UploadTransportError
from avia_cli.core.uploads.source_file import VerifiedSourceFile

_HEADER_NAME = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+\Z")


def _validated_upload_headers(
    headers: dict[str, object],
    *,
    expected_length: int,
) -> dict[str, str]:
    clean: dict[str, str] = {}
    seen: set[str] = set()
    for raw_name, raw_value in dict(headers).items():
        if not isinstance(raw_name, str) or not isinstance(raw_value, str):
            raise RuntimeError("upload required_headers names and values must be strings")
        name = raw_name
        value = raw_value
        if not _HEADER_NAME.fullmatch(name):
            raise RuntimeError("upload required_headers contains an invalid name")
        if value != value.strip() or any(
            ord(character) < 32 or ord(character) == 127 for character in value
        ):
            raise RuntimeError("upload required_headers contains an invalid value")
        try:
            value.encode("latin-1")
        except UnicodeEncodeError as exc:
            raise RuntimeError(
                "upload required_headers value must be ISO-8859-1 encodable"
            ) from exc
        normalized = name.lower()
        if normalized in seen:
            raise RuntimeError(f"upload required_headers duplicates {name} case-insensitively")
        seen.add(normalized)
        if normalized == "host":
            raise RuntimeError("upload required_headers must not override Host")
        if normalized == "transfer-encoding":
            raise RuntimeError("upload required_headers must not use Transfer-Encoding")
        if normalized == "content-length":
            if value != str(expected_length):
                raise RuntimeError(
                    "upload Content-Length does not match the verified source range: "
                    f"expected={expected_length} supplied={value}"
                )
            continue
        clean[name] = value
    clean["Content-Length"] = str(expected_length)
    return clean


def validate_upload_contract(
    *, upload_url: str, headers: dict[str, object], expected_length: int
) -> parse.SplitResult:
    try:
        parsed = parse.urlsplit(upload_url)
        _ = parsed.port
    except ValueError as exc:
        raise RuntimeError("upload URL contract is malformed") from exc
    if (
        not upload_url.isascii()
        or any(character.isspace() or ord(character) < 32 for character in upload_url)
        or parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or bool(parsed.fragment)
    ):
        raise RuntimeError("upload URL contract must be an absolute credential-free http(s) URL")
    _validated_upload_headers(headers, expected_length=expected_length)
    return parsed


def put_file_requests(
    *,
    upload_url: str,
    source: VerifiedSourceFile,
    headers: dict[str, object],
    upload_error: type[RuntimeError],
    connect_timeout: float,
    read_timeout: float,
) -> None:
    import requests

    expected_length = int(source.identity["size_bytes"])
    validate_upload_contract(
        upload_url=upload_url,
        headers=headers,
        expected_length=expected_length,
    )
    if connect_timeout <= 0 or read_timeout <= 0:
        raise ValueError("upload timeouts must be greater than zero")
    request_headers = _validated_upload_headers(headers, expected_length=expected_length)
    handle = source.prepare()
    try:
        with requests.Session() as session:
            resp = session.put(
                upload_url,
                data=handle,
                headers=request_headers,
                timeout=(float(connect_timeout), float(read_timeout)),
                allow_redirects=False,
            )
    except (
        requests.exceptions.InvalidHeader,
        requests.exceptions.InvalidSchema,
        requests.exceptions.InvalidURL,
        requests.exceptions.MissingSchema,
        requests.exceptions.TooManyRedirects,
        requests.exceptions.URLRequired,
    ) as exc:
        source.assert_unchanged(context="source file changed during failed transfer")
        raise RuntimeError(
            f"folder PUT request contract is invalid: {type(exc).__name__}"
        ) from None
    except (
        requests.exceptions.ChunkedEncodingError,
        requests.exceptions.ConnectionError,
        requests.exceptions.ContentDecodingError,
        requests.exceptions.Timeout,
    ) as exc:
        source.assert_unchanged(context="source file changed during failed transfer")
        raise _UploadTransportError("folder PUT") from exc
    except requests.exceptions.RequestException as exc:
        source.assert_unchanged(context="source file changed during failed transfer")
        raise RuntimeError(
            f"folder PUT request failed with non-transport error: {type(exc).__name__}"
        ) from None
    source.assert_position(expected_length)
    source.assert_unchanged()
    if not 200 <= int(resp.status_code) < 300:
        raise upload_error(
            status=int(resp.status_code),
            reason=str(resp.reason or ""),
            detail=resp.text[:500],
        )
