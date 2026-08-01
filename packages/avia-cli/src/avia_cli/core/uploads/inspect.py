from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast
from urllib import parse

from avia_cli.core.uploads.api import _project_path, _request_json_with_retries
from avia_cli.core.uploads.contracts import (
    ANOMALIB_CLASSES,
    require_format_task,
)
from avia_cli.core.uploads.class_catalog import require_canonical_class_catalog
from avia_cli.core.uploads.inventory import require_manifest_inventory
from avia_cli.core.uploads.manifest import ManifestImageError, scan_source_manifest
from avia_cli.core.atomic_file import read_regular_file
from avia_cli.core.strict_json import strict_json_loads
from avia_cli.core.uploads.response_contracts import (
    IMPORT_STATUSES,
    parse_import_manifest_identity,
    require_canonical_import_id,
    validate_version_ref_phase,
)
from avia_cli.core.uploads.state import _safe_state_segment, _validate_state
from avia_cli.core.uploads.validation import inspect_dataset_class_catalog, validate_dataset

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


def _manifest_image_failure(
    *,
    source: str | Path,
    format_name: str,
    task_key: str,
    error: ManifestImageError,
) -> dict[str, Any]:
    expected_format = getattr(error, "expected_format", None)
    actual_format = getattr(error, "actual_format", None)
    details = (
        {
            "expected_format": expected_format,
            "actual_format": actual_format,
        }
        if isinstance(expected_format, str) and isinstance(actual_format, str)
        else {}
    )
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
                "message": str(error),
                "path": str(getattr(error, "relative_path", "")),
                **details,
            }
        ],
        "warnings": [],
    }


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
    except ManifestImageError as exc:
        return _manifest_image_failure(
            source=source,
            format_name=format_name,
            task_key=task_key,
            error=exc,
        )
    summary = _manifest_summary(manifest, format_name=format_name, task_key=task_key)
    classes, errors = inspect_dataset_class_catalog(
        source_root=Path(str(manifest["source"])),
        manifest=manifest,
        format_name=format_name,
        task_key=task_key,
    )
    return {
        **summary,
        "classes": classes,
        "status": "failed" if errors else "ok",
        "error_count": len(errors),
        "warning_count": 0,
        "errors": errors,
        "warnings": [],
    }


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
    except ManifestImageError as exc:
        return _manifest_image_failure(
            source=source,
            format_name=format_name,
            task_key=task_key,
            error=exc,
        )
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
        "sample_files": [str(item.get("relative_path") or "") for item in files[:10]],
    }


def _manifest_classes(manifest: dict[str, object], *, format_name: str) -> list[str]:
    if format_name == "anomalib":
        return list(ANOMALIB_CLASSES)
    classes = manifest.get("classes")
    return require_canonical_class_catalog(
        [] if classes is None else classes,
        label="manifest classes",
        allow_empty=True,
    )


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
    assert isinstance(status, str)
    manifest_identity = parse_import_manifest_identity(
        item["object_key"],
        import_id=item["import_id"],
        label="ingestion-jobs entry object_key",
    )
    validate_version_ref_phase(
        item,
        status=status,
        label="ingestion-jobs entry",
        import_id=manifest_identity.import_id,
        project_scope_id=manifest_identity.project_scope_id,
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
            raw = strict_json_loads(read_regular_file(path).decode("utf-8"))
        except (OSError, UnicodeDecodeError, ValueError) as exc:
            raise SystemExit(f"invalid cleanup state {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise SystemExit(f"invalid cleanup state {path}: expected JSON object")
        try:
            _validate_state(raw, path=path)
        except ValueError as exc:
            raise SystemExit(f"invalid cleanup state {path}: {exc}") from exc
        files = cast(dict[str, dict[str, Any]], raw["files"])
        states.append(
            {
                "project_id": raw["project_id"],
                "import_id": raw["import_id"],
                "source": raw["source"],
                "format": raw["format"],
                "task_key": raw["task_key"],
                "phase": raw["phase"],
                "total_files": len(files),
                "uploaded_files": sum(1 for value in files.values() if value["uploaded"]),
                "streamed_files": sum(1 for value in files.values() if value["streamed"]),
                "state_path": str(path),
            }
        )
    return states


def _build_cleanup_actions(
    *,
    local_states: list[dict[str, Any]],
    server_imports: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    server_by_import: dict[str, dict[str, Any]] = {}
    for item in server_imports:
        _validate_compact_server_import(item)
        import_id = cast(str, item["import_id"])
        if import_id in server_by_import:
            raise RuntimeError(f"cleanup server imports duplicate {import_id}")
        server_by_import[import_id] = item
    actions: list[dict[str, Any]] = []
    for state in local_states:
        _validate_compact_local_state(state)
        import_id = state["import_id"]
        if import_id is None:
            continue
        assert isinstance(import_id, str)
        server = server_by_import.get(import_id)
        status = server["status"] if server is not None else None
        if state["phase"] == "completed" and status in _TERMINAL_STATUSES:
            actions.append(
                {
                    "kind": "remove_local_state",
                    "path": state["state_path"],
                    "reason": "server import is terminal and local resume state is completed",
                }
            )
    return actions


def _validate_compact_server_import(item: object) -> None:
    required = {"import_id", "job_type", "object_key", "status"}
    optional = {"created_at", "dataset_version_id", "progress", "updated_at", "version_ref"}
    if (
        not isinstance(item, dict)
        or not required.issubset(item)
        or not set(item) <= required | optional
    ):
        raise RuntimeError("cleanup server import fields must be exact")
    import_id = item["import_id"]
    try:
        require_canonical_import_id(import_id, label="cleanup server import_id")
    except RuntimeError as exc:
        raise RuntimeError(str(exc)) from exc
    if item["status"] not in IMPORT_STATUSES:
        raise RuntimeError("cleanup server import status must be canonical")
    for field in ("job_type", "object_key"):
        value = item[field]
        if not isinstance(value, str) or not value or value != value.strip():
            raise RuntimeError(f"cleanup server import {field} must be canonical")


def _validate_compact_local_state(state: object) -> None:
    expected = {
        "format",
        "import_id",
        "phase",
        "project_id",
        "source",
        "state_path",
        "streamed_files",
        "task_key",
        "total_files",
        "uploaded_files",
    }
    if not isinstance(state, dict) or set(state) != expected:
        raise RuntimeError("cleanup local state fields must be exact")
    phase = state["phase"]
    if phase not in {"session_pending", "uploading", "completed"}:
        raise RuntimeError("cleanup local state phase must be canonical")
    import_id = state["import_id"]
    if phase == "session_pending":
        if import_id is not None:
            raise RuntimeError("cleanup session_pending state must not have import_id")
    else:
        try:
            require_canonical_import_id(import_id, label="cleanup local import_id")
        except RuntimeError as exc:
            raise RuntimeError(str(exc)) from exc
    for field in ("format", "project_id", "source", "state_path", "task_key"):
        value = state[field]
        if not isinstance(value, str) or not value or value != value.strip():
            raise RuntimeError(f"cleanup local state {field} must be canonical")
    for field in ("streamed_files", "total_files", "uploaded_files"):
        value = state[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"cleanup local state {field} must be non-negative")
    if not 0 <= state["streamed_files"] <= state["uploaded_files"] <= state["total_files"]:
        raise RuntimeError("cleanup local state file counts are inconsistent")
