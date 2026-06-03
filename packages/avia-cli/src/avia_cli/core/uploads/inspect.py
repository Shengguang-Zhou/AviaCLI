from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib import parse

from avia_cli.core.uploads.api import _project_path, _request_json_with_retries
from avia_cli.core.uploads.manifest import _is_image_path, scan_source_manifest
from avia_cli.core.uploads.state import _safe_state_segment

_TERMINAL_STATUSES = {
    "succeeded",
    "success",
    "completed",
    "complete",
    "failed",
    "error",
    "cancelled",
    "canceled",
}
_CLASS_FILES = {"classes.txt", "data.yaml", "data.yml", "dataset.yaml", "dataset.yml"}


def inspect_dataset(
    *,
    source: str | Path,
    format_name: str,
    hash_workers: int = 1,
    max_files: int | None = None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    manifest = scan_source_manifest(
        source,
        include_sha256=False,
        include_dimensions=False,
        hash_workers=hash_workers,
        max_files=max_files,
        max_samples=max_samples,
        format_name=format_name,
    )
    return _manifest_summary(manifest, format_name=format_name)


def verify_dataset(
    *,
    source: str | Path,
    format_name: str,
    hash_workers: int = 1,
    max_files: int | None = None,
    max_samples: int | None = None,
) -> dict[str, Any]:
    manifest = scan_source_manifest(
        source,
        include_sha256=False,
        include_dimensions=False,
        hash_workers=hash_workers,
        max_files=max_files,
        max_samples=max_samples,
        format_name=format_name,
    )
    summary = _manifest_summary(manifest, format_name=format_name)
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    if str(format_name).lower() == "yolo":
        _verify_yolo_manifest(
            source_root=Path(str(manifest["source"])),
            manifest=manifest,
            errors=errors,
            warnings=warnings,
        )
    elif int(summary["image_count"]) == 0:
        errors.append({"code": "no_images", "message": "dataset has no image files"})
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


def _manifest_summary(manifest: dict[str, object], *, format_name: str) -> dict[str, Any]:
    files = [dict(item) for item in list(manifest.get("files") or []) if isinstance(item, dict)]
    image_paths = [
        str(item.get("relative_path") or "")
        for item in files
        if _is_image_path(str(item.get("relative_path") or ""))
    ]
    label_paths = [
        str(item.get("relative_path") or "")
        for item in files
        if str(item.get("relative_path") or "").startswith("labels/")
        and str(item.get("relative_path") or "").endswith(".txt")
    ]
    return {
        "source": str(manifest["source"]),
        "format": str(format_name).lower(),
        "file_count": int(manifest.get("file_count") or 0),
        "total_bytes": int(manifest.get("total_bytes") or 0),
        "image_count": len(image_paths),
        "label_count": len(label_paths),
        "classes": [str(item) for item in list(manifest.get("classes") or [])],
        "sample_files": [str(item.get("relative_path") or "") for item in files[:10]],
    }


def _verify_yolo_manifest(
    *,
    source_root: Path,
    manifest: dict[str, object],
    errors: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> None:
    files = [dict(item) for item in list(manifest.get("files") or []) if isinstance(item, dict)]
    relative_paths = {str(item.get("relative_path") or "") for item in files}
    image_paths = sorted(
        path for path in relative_paths if path.startswith("images/") and _is_image_path(path)
    )
    label_paths = sorted(
        path for path in relative_paths if path.startswith("labels/") and path.endswith(".txt")
    )
    classes = [str(item) for item in list(manifest.get("classes") or [])]
    if not image_paths:
        errors.append({"code": "no_images", "message": "YOLO dataset has no images/"})
    if not classes and not (relative_paths & _CLASS_FILES):
        warnings.append(
            {
                "code": "missing_class_names",
                "message": "YOLO dataset has no classes.txt or data.yaml class names",
            }
        )
    expected_labels = {_label_path_for_image(path) for path in image_paths}
    for image_path in image_paths:
        expected = _label_path_for_image(image_path)
        if expected not in relative_paths:
            warnings.append(
                {
                    "code": "missing_yolo_label",
                    "path": image_path,
                    "expected_label_path": expected,
                    "message": "image has no matching YOLO label file",
                }
            )
    for label_path in label_paths:
        if label_path not in expected_labels:
            warnings.append(
                {
                    "code": "orphan_yolo_label",
                    "path": label_path,
                    "message": "label has no matching image file",
                }
            )
        _verify_yolo_label_file(
            path=source_root / label_path,
            relative_path=label_path,
            class_count=len(classes),
            errors=errors,
            warnings=warnings,
        )


def _verify_yolo_label_file(
    *,
    path: Path,
    relative_path: str,
    class_count: int,
    errors: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        errors.append(
            {
                "code": "label_read_failed",
                "path": relative_path,
                "message": str(exc),
            }
        )
        return
    for line_number, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            errors.append(
                {
                    "code": "invalid_yolo_label",
                    "path": relative_path,
                    "line": line_number,
                    "message": "YOLO label rows must contain class plus coordinates",
                }
            )
            continue
        try:
            class_id = int(float(parts[0]))
            [float(item) for item in parts[1:]]
        except ValueError:
            errors.append(
                {
                    "code": "invalid_yolo_label",
                    "path": relative_path,
                    "line": line_number,
                    "message": "YOLO label rows must be numeric",
                }
            )
            continue
        if class_id < 0:
            errors.append(
                {
                    "code": "invalid_yolo_class",
                    "path": relative_path,
                    "line": line_number,
                    "message": "YOLO class id must be non-negative",
                }
            )
        elif class_count and class_id >= class_count:
            warnings.append(
                {
                    "code": "unknown_yolo_class",
                    "path": relative_path,
                    "line": line_number,
                    "class_id": class_id,
                    "message": "YOLO class id is outside declared class names",
                }
            )


def _label_path_for_image(image_path: str) -> str:
    stem = Path(image_path).with_suffix("").as_posix()
    if stem.startswith("images/"):
        return f"labels/{stem[len('images/') :]}.txt"
    return f"{stem}.txt"


def _list_server_imports(
    *,
    api: str,
    token: object,
    project_id: str,
    limit: int,
) -> list[dict[str, Any]]:
    query = parse.urlencode({"limit": max(1, min(200, int(limit or 50)))})
    response = _request_json_with_retries(
        method="GET",
        url=f"{_project_path(api, project_id, 'ingestion-jobs')}?{query}",
        token=token,
        timeout=60,
        retries=2,
        label="cleanup-plan",
    )
    return [
        _compact_server_import(dict(item))
        for item in list(response.get("imports") or [])
        if isinstance(item, dict)
    ]


def _compact_server_import(item: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {
        "import_id": str(item.get("import_id") or ""),
        "status": str(item.get("status") or ""),
    }
    for key in ("source", "format", "file_count", "image_count", "created_at", "updated_at"):
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
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(raw, dict):
            continue
        files = dict(raw.get("files") or {})
        states.append(
            {
                "project_id": str(raw.get("project_id") or ""),
                "import_id": str(raw.get("import_id") or ""),
                "source": str(raw.get("source") or ""),
                "format": str(raw.get("format") or ""),
                "completed": bool(raw.get("completed")),
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
        if bool(state.get("completed")) and status in _TERMINAL_STATUSES:
            actions.append(
                {
                    "kind": "remove_local_state",
                    "path": str(state.get("state_path") or ""),
                    "reason": "server import is terminal and local resume state is completed",
                }
            )
    return actions
