from __future__ import annotations

from typing import Any


_REF_KEYS = ("dataset_manifest_ref", "artifact_ref", "result_manifest_ref")


def _dict_at(value: object, *path: str) -> dict[str, Any]:
    current: object = value
    for key in path:
        if not isinstance(current, dict):
            return {}
        current = current.get(key)
    return dict(current) if isinstance(current, dict) else {}


def attach_upload_refs(result: dict[str, Any]) -> dict[str, Any]:
    sources = (
        _dict_at(result, "complete"),
        _dict_at(result, "multipart", "complete"),
        _dict_at(result, "job"),
    )
    for key in _REF_KEYS:
        if isinstance(result.get(key), dict):
            continue
        for source in sources:
            ref = source.get(key)
            if isinstance(ref, dict):
                result[key] = dict(ref)
                break

    if not isinstance(result.get("read_lease"), dict):
        for source in sources:
            lease = source.get("read_lease")
            if isinstance(lease, dict):
                result["read_lease"] = dict(lease)
                break

    return result
