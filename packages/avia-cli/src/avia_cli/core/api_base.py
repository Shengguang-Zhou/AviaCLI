from __future__ import annotations

import ipaddress
from urllib.parse import urlsplit, urlunsplit


def canonical_api_base(value: object) -> str:
    raw = str(value or "")
    if not raw:
        raise ValueError("API base is required")
    if raw != raw.strip() or any(character.isspace() for character in raw):
        raise ValueError("API base must not contain whitespace")
    try:
        parsed = urlsplit(raw)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("API base is malformed") from exc
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("API base must use http or https")
    if not parsed.hostname:
        raise ValueError("API base must be absolute and include a host")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("API base must not contain user information")
    if parsed.query or parsed.fragment:
        raise ValueError("API base must not contain a query or fragment")
    if not parsed.path.startswith("/") or "//" in parsed.path:
        raise ValueError("API base path must be an absolute canonical path")
    path_segments = parsed.path.split("/")[1:]
    if any(segment in {".", ".."} for segment in path_segments):
        raise ValueError("API base path must not contain dot segments")

    hostname = parsed.hostname.lower()
    try:
        is_ipv6 = isinstance(ipaddress.ip_address(hostname), ipaddress.IPv6Address)
    except ValueError:
        is_ipv6 = False
    host = f"[{hostname}]" if is_ipv6 else hostname
    default_port = 80 if parsed.scheme == "http" else 443
    if port is not None and port != default_port:
        host = f"{host}:{port}"
    path = parsed.path.rstrip("/") or "/"
    canonical = urlunsplit((parsed.scheme, host, path, "", ""))
    if canonical != raw:
        raise ValueError(f"API base is not canonical; use {canonical}")
    return canonical
