from __future__ import annotations

import re
from dataclasses import dataclass
from urllib import parse

from avia_cli.core.errors import _UploadTransportError
from avia_cli.core.uploads.media_types import require_canonical_media_type
from avia_cli.core.uploads.source_file import VerifiedSourceFile

_HEADER_NAME = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+\Z")


@dataclass(frozen=True, slots=True)
class UploadTransportRoute:
    upload_url: str
    proxy_items: tuple[tuple[str, str | None], ...]

    def request_proxies(self) -> dict[str, str | None]:
        return dict(self.proxy_items)


def _resolved_upload_proxies(upload_url: str) -> dict[str, str | None]:
    """Freeze the requests environment route for one validated upload URL."""

    import requests

    environment = requests.utils.get_environ_proxies(upload_url)
    if requests.utils.select_proxy(upload_url, environment) is None:
        return {"http": None, "https": None, "all": None}
    return {str(key): str(value) for key, value in environment.items()}


def resolve_upload_route(upload_url: str) -> UploadTransportRoute:
    """Validate one signed URL and freeze the route shared by its probe and PUT."""

    validate_upload_contract(upload_url=upload_url, headers={}, expected_length=0)
    proxies = _resolved_upload_proxies(upload_url)
    return UploadTransportRoute(
        upload_url=upload_url,
        proxy_items=tuple(sorted(proxies.items())),
    )


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
        normalized = name.lower()
        if normalized == "content-type":
            require_canonical_media_type(value, label="upload Content-Type")
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
    route: UploadTransportRoute,
    source: VerifiedSourceFile,
    headers: dict[str, object],
    upload_error: type[RuntimeError],
    connect_timeout: float,
    read_timeout: float,
) -> str:
    import requests

    expected_length = int(source.identity["size_bytes"])
    validate_upload_contract(
        upload_url=route.upload_url,
        headers=headers,
        expected_length=expected_length,
    )
    if connect_timeout <= 0 or read_timeout <= 0:
        raise ValueError("upload timeouts must be greater than zero")
    request_headers = _validated_upload_headers(headers, expected_length=expected_length)
    handle = source.prepare()
    request_body = b"" if expected_length == 0 else handle
    try:
        with requests.Session() as session:
            session.trust_env = False
            resp = session.put(
                route.upload_url,
                data=request_body,
                headers=request_headers,
                timeout=(float(connect_timeout), float(read_timeout)),
                allow_redirects=False,
                proxies=route.request_proxies(),
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
    version_id = str(resp.headers.get("x-amz-version-id") or "").strip()
    if not version_id or version_id == "null":
        raise RuntimeError("folder PUT response did not include an S3 VersionId")
    return version_id
