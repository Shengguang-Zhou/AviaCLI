from __future__ import annotations

import re

_MEDIA_TYPE_TOKEN = re.compile(r"[!#$%&'*+\-.^_`|~0-9a-z]+\Z")


def require_canonical_media_type(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.isascii() or value != value.lower():
        raise RuntimeError(f"{label} must be a canonical lowercase ASCII media type")
    if value.count("/") != 1:
        raise RuntimeError(f"{label} must contain exactly one type/subtype separator")
    type_token, subtype_token = value.split("/", 1)
    if (
        _MEDIA_TYPE_TOKEN.fullmatch(type_token) is None
        or _MEDIA_TYPE_TOKEN.fullmatch(subtype_token) is None
    ):
        raise RuntimeError(f"{label} must use non-empty HTTP token type and subtype")
    return value
