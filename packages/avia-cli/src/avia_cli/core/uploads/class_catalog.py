"""Cycle-free canonical ordered class-catalog identity."""

from __future__ import annotations

import re
import unicodedata

_CANONICAL_DECIMAL_INDEX = re.compile(r"(?:0|[1-9][0-9]*)\Z")
MAX_CLASS_NAME_CODEPOINTS = 200
MAX_CLASS_CATALOG_SIZE = 10_000
MAX_CLASS_ID = MAX_CLASS_CATALOG_SIZE - 1


def require_canonical_class_catalog(
    value: object,
    *,
    label: str,
    allow_empty: bool = False,
) -> list[str]:
    if type(value) is not list:
        raise ValueError(f"{label} must be an array")
    if not value and not allow_empty:
        raise ValueError(f"{label} must not be empty")
    if len(value) > MAX_CLASS_CATALOG_SIZE:
        raise ValueError(f"{label} must contain at most {MAX_CLASS_CATALOG_SIZE} classes")

    names: list[str] = []
    for index, name in enumerate(value):
        if (
            type(name) is not str
            or not name
            or len(name) > MAX_CLASS_NAME_CODEPOINTS
            or name != name.strip()
            or name != unicodedata.normalize("NFC", name)
            or any(unicodedata.category(character) == "Cc" for character in name)
        ):
            raise ValueError(
                f"{label}[{index}] must be a non-empty NFC string of at most "
                f"{MAX_CLASS_NAME_CODEPOINTS} Unicode code points without whitespace "
                "padding or control characters"
            )
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError(f"{label} must contain unique class names")
    return names


def require_canonical_class_index(value: object, *, label: str) -> int:
    if type(value) is int:
        index = value
    elif type(value) is str and _CANONICAL_DECIMAL_INDEX.fullmatch(value) is not None:
        index = int(value)
    else:
        raise ValueError(f"{label} must be a non-negative integer or canonical decimal string")
    if index < 0 or index > MAX_CLASS_ID:
        raise ValueError(f"{label} must be between 0 and {MAX_CLASS_ID}")
    return index


def require_indexed_class_catalog(value: object, *, label: str) -> list[str]:
    if type(value) is list:
        return require_canonical_class_catalog(value, label=label)
    if type(value) is not dict:
        raise ValueError(f"{label} must be a list or integer-indexed mapping")
    if not value:
        raise ValueError(f"{label} must not be empty")
    if len(value) > MAX_CLASS_CATALOG_SIZE:
        raise ValueError(f"{label} must contain at most {MAX_CLASS_CATALOG_SIZE} classes")

    indexed: list[tuple[int, object]] = []
    seen: set[int] = set()
    for raw_index, name in value.items():
        index = require_canonical_class_index(
            raw_index,
            label=f"{label} index {raw_index!r}",
        )
        if index in seen:
            raise ValueError(f"{label} indices must be unique")
        seen.add(index)
        indexed.append((index, name))
    indexed.sort(key=lambda item: item[0])
    if [index for index, _name in indexed] != list(range(len(indexed))):
        raise ValueError(f"{label} indices must be contiguous from zero")
    return require_canonical_class_catalog(
        [name for _index, name in indexed],
        label=label,
    )


def require_class_count(value: object, *, label: str, allow_zero: bool) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be a {qualifier} integer")
    if value > MAX_CLASS_CATALOG_SIZE:
        raise ValueError(f"{label} must be at most {MAX_CLASS_CATALOG_SIZE}")
    return value


__all__ = [
    "MAX_CLASS_CATALOG_SIZE",
    "MAX_CLASS_ID",
    "MAX_CLASS_NAME_CODEPOINTS",
    "require_canonical_class_catalog",
    "require_canonical_class_index",
    "require_class_count",
    "require_indexed_class_catalog",
]
