from __future__ import annotations

from urllib import parse


def absolute_url_from_api(api: str, raw_url: str) -> str:
    text = str(raw_url or "").strip()
    if not text:
        return text
    parsed = parse.urlsplit(text)
    if parsed.scheme in {"http", "https"}:
        return text
    if not text.startswith("/"):
        return text
    api_parsed = parse.urlsplit(str(api or "").strip())
    if api_parsed.scheme not in {"http", "https"} or not api_parsed.netloc:
        raise RuntimeError(f"cannot resolve relative upload URL without absolute API URL: {api}")
    return f"{api_parsed.scheme}://{api_parsed.netloc}{text}"


def upload_url_from_api(api: str, raw_url: str) -> str:
    return absolute_url_from_api(api, raw_url)


def upload_request_from_api(
    api: str,
    raw_url: str,
    *,
    upload_url_origin_override: str = "",
) -> tuple[str, dict[str, str]]:
    url = upload_url_from_api(api, raw_url)
    override = str(upload_url_origin_override or "").strip().rstrip("/")
    if not override:
        return url, {}
    parsed = parse.urlsplit(url)
    override_parsed = parse.urlsplit(override)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise RuntimeError(f"cannot override non-absolute upload URL: {raw_url}")
    if override_parsed.scheme not in {"http", "https"} or not override_parsed.netloc:
        raise RuntimeError(
            f"upload URL origin override must be absolute http(s): {upload_url_origin_override}"
        )
    rewritten = parse.urlunsplit(
        (
            override_parsed.scheme,
            override_parsed.netloc,
            parsed.path,
            parsed.query,
            parsed.fragment,
        )
    )
    return rewritten, {"Host": parsed.netloc}
