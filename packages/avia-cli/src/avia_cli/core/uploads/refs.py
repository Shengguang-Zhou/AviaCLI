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
        _dict_at(result, "job"),
    )
    for key in _REF_KEYS:
        candidates = [result.get(key), *(source.get(key) for source in sources)]
        refs = [dict(item) for item in candidates if isinstance(item, dict)]
        if any(item is not None and not isinstance(item, dict) for item in candidates):
            raise RuntimeError(f"upload response {key} must be an object when present")
        if refs and any(ref != refs[0] for ref in refs[1:]):
            raise RuntimeError(f"upload response contains conflicting {key} identities")
        if refs:
            result[key] = refs[0]

    candidates = [result.get("read_lease"), *(source.get("read_lease") for source in sources)]
    leases = [dict(item) for item in candidates if isinstance(item, dict)]
    if any(item is not None and not isinstance(item, dict) for item in candidates):
        raise RuntimeError("upload response read_lease must be an object when present")
    if leases and any(lease != leases[0] for lease in leases[1:]):
        raise RuntimeError("upload response contains conflicting read_lease identities")
    if leases:
        result["read_lease"] = leases[0]

    return result
