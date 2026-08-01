from __future__ import annotations

import json
import math
from typing import Any


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"JSON number is outside the finite float range: {value}")
    return parsed


def strict_json_loads(document: str | bytes | bytearray) -> Any:
    """Decode RFC JSON while rejecting ambiguous objects and non-finite numbers."""

    return json.loads(
        document,
        object_pairs_hook=_unique_object,
        parse_constant=_reject_nonfinite_constant,
        parse_float=_finite_float,
    )


__all__ = ["strict_json_loads"]
