from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib import parse

from avia_cli.core.uploads.api import _project_path, _request_json_with_retries
from avia_cli.core.uploads.contracts import ANOMALIB_CLASSES, require_format_task
from avia_cli.core.uploads.inventory import require_manifest_inventory
from avia_cli.core.uploads.manifest import scan_source_manifest
from avia_cli.core.atomic_file import read_regular_file
from avia_cli.core.uploads.response_contracts import (
    IMPORT_STATUSES,
    validate_version_ref_phase,
)
from avia_cli.core.uploads.state import _safe_state_segment, _validate_state
from avia_cli.core.uploads.validation import validate_dataset

_TERMINAL_STATUSES = {"succeeded", "failed"}
_INGESTION_JOB_BASE_FIELDS = {
    "created_at",
    "dataset_validation",
    "error",
    "import_id",
    "job_type",
    "object_key",
    "progress",
    "status",
    "updated_at",
}
_INGESTION_JOB_IDENTITY_FIELDS = {"dataset_version_id", "version_ref"}


def inspect_dataset(
    *,
    source: str | Path,
    format_name: str,
    task_key: str,
    hash_workers: int = 1,
) -> dict[str, Any]:
    format_name, task_key = require_format_task(format_name=format_name, task_key=task_key)
    try:
        manifest = scan_source_manifest(
            source,
            include_dimensions=False,
            hash_workers=hash_workers,
            format_name=format_name,
        )
    except RuntimeError as exc:
        return {
            "source": str(Path(source).expanduser().resolve()),
            "format": format_name,
            "task_key": task_key,
            "file_count": 0,
            "image_count": 0,
            "label_count": 0,
            "mask_count": 0,
            "total_bytes": 0,
            "classes": _manifest_classes({}, format_name=format_name),
            "status": "failed",
            "error_count": 1,
            "warning_count": 0,
            "errors": [
                {
                    "code": "invalid_image",
                    "message": str(exc),
                    "path": str(getattr(exc, "relative_path", "")),
                }
            ],
            "warnings": [],
        }
    return _manifest_summary(manifest, format_name=format_name, task_key=task_key)


def verify_dataset(
    *,
    source: str | Path,
    format_name: str,
    task_key: str,
    hash_workers: int = 1,
) -> dict[str, Any]:
    format_name, task_key = require_format_task(format_name=format_name, task_key=task_key)
    try:
        manifest = scan_source_manifest(
            source,
            include_dimensions=False,
            hash_workers=hash_workers,
            format_name=format_name,
        )
    except RuntimeError as exc:
        return {
            "source": str(Path(source).expanduser().resolve()),
            "format": format_name,
            "task_key": task_key,
            "file_count": 0,
            "image_count": 0,
            "label_count": 0,
            "mask_count": 0,
            "total_bytes": 0,
            "classes": _manifest_classes({}, format_name=format_name),
            "status": "failed",
            "error_count": 1,
            "warning_count": 0,
            "errors": [
                {
                    "code": "invalid_image",
                    "message": str(exc),
                    "path": str(getattr(exc, "relative_path", "")),
                }
            ],
            "warnings": [],
        }
    summary = _manifest_summary(manifest, format_name=format_name, task_key=task_key)
    classes, errors, warnings = validate_dataset(
        source_root=Path(str(manifest["source"])),
        manifest=manifest,
        format_name=format_name,
        task_key=task_key,
    )
    summary["classes"] = classes
    result = {
        **summary,
        "status": "failed" if errors else "ok",
        "error_count": len(errors),
        "warning_count": len(warnings),
        "errors": errors,
        "warnings": warnings,
    }
    return result


def build_cleanup_plan(
    *,
    api: str,
    token: object,
    project_id: str,
    source: str | Path | None = None,
    state_dir: str | Path | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    server_imports = _list_server_imports(
        api=api,
        token=token,
        project_id=project_id,
        limit=limit,
    )
    local_state_dir = _resolve_state_dir(source=source, state_dir=state_dir)
    local_states = _load_local_states(
        local_state_dir=local_state_dir,
        project_id=project_id,
    )
    actions = _build_cleanup_actions(
        local_states=local_states,
        server_imports=server_imports,
    )
    return {
        "project_id": str(project_id),
        "source": str(Path(source).expanduser().resolve()) if source else None,
        "local_state_dir": str(local_state_dir) if local_state_dir else None,
        "storage_boundary": "server_owned",
        "server_imports": server_imports,
        "local_states": local_states,
        "actions": actions,
        "notes": [
            "AviaCLI does not connect to MinIO or lakeFS directly.",
            "Server-side object and version cleanup must be performed by YoloTaskCV.",
        ],
    }


def _manifest_summary(
    manifest: dict[str, object], *, format_name: str, task_key: str
) -> dict[str, Any]:
    files = [dict(item) for item in list(manifest.get("files") or []) if isinstance(item, dict)]
    inventory = require_manifest_inventory(manifest, format_name=format_name)
    return {
        "source": str(manifest["source"]),
        "format": format_name,
        "task_key": task_key,
        "file_count": int(manifest["file_count"]),
        "total_bytes": int(manifest["total_bytes"]),
        **inventory.counts(),
        "classes": _manifest_classes(manifest, format_name=format_name),
        "sample_files": [str(item.get("relative_path") or "") for item in files[:10]],
    }


def _manifest_classes(manifest: dict[str, object], *, format_name: str) -> list[str]:
    if format_name == "anomalib":
        return list(ANOMALIB_CLASSES)
    return [str(item) for item in list(manifest.get("classes") or [])]


def _list_server_imports(
    *,
    api: str,
    token: object,
    project_id: str,
    limit: int,
) -> list[dict[str, Any]]:
    parsed_limit = int(limit)
    if parsed_limit < 1 or parsed_limit > 200:
        raise ValueError("cleanup plan limit must be between 1 and 200")
    query = parse.urlencode({"limit": parsed_limit})
    response = _request_json_with_retries(
        method="GET",
        url=f"{_project_path(api, project_id, 'ingestion-jobs')}?{query}",
        token=token,
        timeout=60,
        retries=2,
        label="cleanup-plan",
    )
    if set(response) != {"imports", "next_cursor", "project_id"}:
        raise RuntimeError("ingestion-jobs response fields must be exact")
    if response.get("project_id") != project_id or not isinstance(response.get("imports"), list):
        raise RuntimeError("ingestion-jobs response identity does not match the request")
    next_cursor = response.get("next_cursor")
    if next_cursor is not None and (
        not isinstance(next_cursor, str) or not next_cursor or next_cursor != next_cursor.strip()
    ):
        raise RuntimeError("ingestion-jobs next_cursor must be null or a canonical string")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in response["imports"]:
        if not isinstance(item, dict):
            raise RuntimeError("ingestion-jobs entries must be objects")
        _validate_server_import(item)
        import_id = item["import_id"]
        status = item.get("status")
        if not import_id or import_id in seen or status not in IMPORT_STATUSES:
            raise RuntimeError("ingestion-jobs entries require unique ids and canonical statuses")
        seen.add(import_id)
        result.append(_compact_server_import(item))
    return result


def _validate_server_import(item: dict[str, Any]) -> None:
    actual_fields = set(item)
    expected_fields = (
        _INGESTION_JOB_BASE_FIELDS | _INGESTION_JOB_IDENTITY_FIELDS
        if _INGESTION_JOB_IDENTITY_FIELDS.issubset(actual_fields)
        else _INGESTION_JOB_BASE_FIELDS
    )
    if actual_fields != expected_fields:
        raise RuntimeError(
            "ingestion-jobs entry fields must be exact: "
            f"expected={sorted(expected_fields)} actual={sorted(item)}"
        )
    for key in ("import_id", "job_type", "object_key"):
        value = item.get(key)
        if not isinstance(value, str) or not value or value != value.strip():
            raise RuntimeError(f"ingestion-jobs entry {key} must be a canonical non-empty string")
    for key in ("progress", "error"):
        if not isinstance(item.get(key), dict):
            raise RuntimeError(f"ingestion-jobs entry {key} must be an object")
    dataset_validation = item.get("dataset_validation")
    if dataset_validation is not None and not isinstance(dataset_validation, dict):
        raise RuntimeError("ingestion-jobs entry dataset_validation must be null or an object")
    status = item.get("status")
    if status not in IMPORT_STATUSES:
        raise RuntimeError(f"ingestion-jobs entry has unsupported status: {status!r}")
    validate_version_ref_phase(
        item,
        status=str(status),
        label="ingestion-jobs entry",
    )
    for key in ("created_at", "updated_at"):
        value = item.get(key)
        if value is not None and (
            not isinstance(value, str) or not value or value != value.strip()
        ):
            raise RuntimeError(f"ingestion-jobs entry {key} must be null or a canonical string")


def _compact_server_import(item: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {
        "import_id": item["import_id"],
        "status": item["status"],
        "job_type": item["job_type"],
        "object_key": item["object_key"],
    }
    for key in ("dataset_version_id", "version_ref", "created_at", "updated_at"):
        if key in item:
            compact[key] = item[key]
    progress = item.get("progress")
    if isinstance(progress, dict):
        compact["progress"] = {
            key: value
            for key, value in progress.items()
            if key in {"phase", "status", "image_count", "file_count", "items_total"}
        }
    return compact


def _resolve_state_dir(
    *,
    source: str | Path | None,
    state_dir: str | Path | None,
) -> Path | None:
    explicit = str(state_dir or os.environ.get("AVIA_STATE_DIR", "")).strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    if source is None:
        return None
    return Path(source).expanduser().resolve() / ".avia" / "imports"


def _load_local_states(
    *,
    local_state_dir: Path | None,
    project_id: str,
) -> list[dict[str, Any]]:
    if local_state_dir is None:
        return []
    project_dir = local_state_dir / _safe_state_segment(project_id)
    if not project_dir.exists():
        return []
    states: list[dict[str, Any]] = []
    for path in sorted(project_dir.glob("*.json")):
        try:
            raw = json.loads(read_regular_file(path).decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SystemExit(f"invalid cleanup state {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise SystemExit(f"invalid cleanup state {path}: expected JSON object")
        try:
            _validate_state(raw, path=path)
        except ValueError as exc:
            raise SystemExit(f"invalid cleanup state {path}: {exc}") from exc
        files = dict(raw.get("files") or {})
        states.append(
            {
                "project_id": str(raw.get("project_id") or ""),
                "import_id": str(raw.get("import_id") or ""),
                "source": str(raw.get("source") or ""),
                "format": str(raw.get("format") or ""),
                "task_key": str(raw.get("task_key") or ""),
                "phase": str(raw["phase"]),
                "total_files": len(files),
                "uploaded_files": sum(
                    1
                    for value in files.values()
                    if isinstance(value, dict) and bool(value.get("uploaded"))
                ),
                "streamed_files": sum(
                    1
                    for value in files.values()
                    if isinstance(value, dict) and bool(value.get("streamed"))
                ),
                "state_path": str(path),
            }
        )
    return states


def _build_cleanup_actions(
    *,
    local_states: list[dict[str, Any]],
    server_imports: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    server_by_import = {
        str(item.get("import_id") or ""): dict(item)
        for item in server_imports
        if str(item.get("import_id") or "")
    }
    actions: list[dict[str, Any]] = []
    for state in local_states:
        import_id = str(state.get("import_id") or "")
        status = str(server_by_import.get(import_id, {}).get("status") or "").lower()
        if state.get("phase") == "completed" and status in _TERMINAL_STATUSES:
            actions.append(
                {
                    "kind": "remove_local_state",
                    "path": str(state.get("state_path") or ""),
                    "reason": "server import is terminal and local resume state is completed",
                }
            )
    return actions
