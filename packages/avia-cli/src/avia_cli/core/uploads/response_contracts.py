from __future__ import annotations

from typing import Any

from avia_cli.core.uploads.contracts import require_format_task, require_object_prefix_uri

IMPORT_ACTIVE_STATUSES = frozenset({"pending_upload", "uploaded", "queued", "running"})
IMPORT_TERMINAL_STATUSES = frozenset({"succeeded", "failed"})
IMPORT_STATUSES = IMPORT_ACTIVE_STATUSES | IMPORT_TERMINAL_STATUSES

_SOURCE_IMPORT_PROGRESS_FIELDS = {
    "all_referenced_existing",
    "classes",
    "file_count",
    "format",
    "manifest_object_key",
    "phase",
    "source_kind",
    "source_owned",
    "source_uri",
    "task_key",
    "total_bytes",
}
_SOURCE_IMPORT_REQUEST_FIELDS = {
    "auto_post_processing",
    "classes",
    "format",
    "source_kind",
    "task_key",
    "uri",
}


def validate_source_import_request(payload: dict[str, object]) -> None:
    _require_exact_fields(payload, _SOURCE_IMPORT_REQUEST_FIELDS, label="source-import request")
    if payload.get("source_kind") != "object_prefix":
        raise RuntimeError("source-import request source_kind must be object_prefix")
    try:
        require_object_prefix_uri(payload.get("uri"))
        require_format_task(
            format_name=str(payload.get("format")),
            task_key=str(payload.get("task_key")),
        )
    except SystemExit as exc:
        raise RuntimeError(str(exc)) from exc
    classes = payload.get("classes")
    if (
        not isinstance(classes, list)
        or any(not isinstance(item, str) or not item or item != item.strip() for item in classes)
        or len(set(classes)) != len(classes)
    ):
        raise RuntimeError("source-import request classes must be unique canonical strings")
    if payload.get("format") != "yolo" and classes:
        raise RuntimeError("source-import request classes are only valid for yolo format")
    if not isinstance(payload.get("auto_post_processing"), bool):
        raise RuntimeError("source-import request auto_post_processing must be boolean")


def decode_source_import_response(
    payload: dict[str, Any],
    *,
    project_id: str,
    request_payload: dict[str, object],
) -> dict[str, Any]:
    validate_source_import_request(request_payload)
    auto_post_processing = request_payload.get("auto_post_processing")
    expected_fields = {
        "import_id",
        "object_key",
        "progress",
        "project_id",
        "status",
        "workspace_id",
    }
    if auto_post_processing:
        expected_fields.update({"dispatch_mode", "reason", "worker_task_id"})
    _require_exact_fields(payload, expected_fields, label="source-import response")
    _require_identity(payload, project_id=project_id)
    expected_status = "queued" if auto_post_processing else "uploaded"
    if payload.get("status") != expected_status:
        raise RuntimeError(f"source-import status must be {expected_status}")
    object_key = _require_nonempty_string(payload, "object_key", label="source-import response")
    progress = payload.get("progress")
    if not isinstance(progress, dict):
        raise RuntimeError("source-import response progress must be an object")
    _require_exact_fields(progress, _SOURCE_IMPORT_PROGRESS_FIELDS, label="source-import progress")
    for key in ("source_kind", "format", "task_key", "classes"):
        if progress.get(key) != request_payload.get(key):
            raise RuntimeError(f"source-import progress {key} does not match the request")
    request_uri = _require_nonempty_string(request_payload, "uri", label="source-import request")
    if progress.get("source_uri") != request_uri:
        raise RuntimeError("source-import progress source_uri does not match the request")
    if progress.get("manifest_object_key") != object_key:
        raise RuntimeError("source-import manifest object identity is inconsistent")
    for key in ("file_count", "total_bytes"):
        value = progress.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeError(f"source-import progress {key} must be a positive integer")
    if progress.get("source_owned") is not False:
        raise RuntimeError("object-prefix source imports must remain reference-owned")
    if not isinstance(progress.get("all_referenced_existing"), bool):
        raise RuntimeError("source-import progress all_referenced_existing must be boolean")
    if progress.get("phase") != "uploaded":
        raise RuntimeError("source-import progress phase must be uploaded")
    if auto_post_processing:
        if payload.get("reason") != "queued":
            raise RuntimeError("source-import reason must be queued")
        _require_nonempty_string(payload, "dispatch_mode", label="source-import response")
        _require_nonempty_string(payload, "worker_task_id", label="source-import response")
    return payload


def decode_dataset_session_response(payload: dict[str, Any], *, project_id: str) -> dict[str, Any]:
    _require_exact_fields(
        payload,
        {
            "dataset_manifest_ref",
            "import_id",
            "object_key",
            "project_id",
            "read_lease",
            "status",
            "workspace_id",
        },
        label="dataset-session response",
    )
    _require_identity(payload, project_id=project_id)
    if payload["status"] != "pending_upload":
        raise RuntimeError("dataset-session status must be pending_upload")
    _require_nonempty_string(payload, "object_key", label="dataset-session response")
    _require_object(payload, "dataset_manifest_ref", label="dataset-session response")
    _require_object(payload, "read_lease", label="dataset-session response")
    return payload


def decode_batch_upload_urls_response(
    payload: dict[str, Any],
    *,
    project_id: str,
    import_id: str,
    requested_files: list[dict[str, object]],
) -> dict[str, Any]:
    _require_exact_fields(
        payload,
        {"files", "import_id", "project_id", "workspace_id"},
        label="batch-upload-urls response",
    )
    _require_identity(payload, project_id=project_id, import_id=import_id)
    raw_files = payload.get("files")
    if not isinstance(raw_files, list):
        raise RuntimeError("batch-upload-urls response files must be an array")
    requested = _unique_requested_files(requested_files)
    returned: dict[str, dict[str, Any]] = {}
    object_keys: set[str] = set()
    upload_urls: set[str] = set()
    exact_fields = {
        "content_type",
        "expires_in",
        "object_key",
        "relative_path",
        "required_headers",
        "sha256",
        "size_bytes",
        "upload_url",
    }
    for raw in raw_files:
        if not isinstance(raw, dict):
            raise RuntimeError("batch-upload-urls response file must be an object")
        _require_exact_fields(raw, exact_fields, label="batch-upload-urls file")
        relative_path = _require_nonempty_string(
            raw, "relative_path", label="batch-upload-urls file"
        )
        if relative_path in returned:
            raise RuntimeError(f"batch-upload-urls response duplicates {relative_path}")
        expected = requested.get(relative_path)
        if expected is None:
            raise RuntimeError(f"batch-upload-urls response returned unrequested {relative_path}")
        if raw.get("size_bytes") != expected.get("size_bytes"):
            raise RuntimeError(f"batch-upload-urls size mismatch for {relative_path}")
        if raw.get("content_type") != expected.get("content_type"):
            raise RuntimeError(f"batch-upload-urls content type mismatch for {relative_path}")
        expected_sha = str(expected.get("sha256") or "") or None
        if raw.get("sha256") != expected_sha:
            raise RuntimeError(f"batch-upload-urls sha256 mismatch for {relative_path}")
        object_key = _require_nonempty_string(raw, "object_key", label="batch-upload-urls file")
        upload_url = _require_nonempty_string(raw, "upload_url", label="batch-upload-urls file")
        if object_key in object_keys or upload_url in upload_urls:
            raise RuntimeError("batch-upload-urls remote object identities must be unique")
        object_keys.add(object_key)
        upload_urls.add(upload_url)
        if not isinstance(raw.get("required_headers"), dict):
            raise RuntimeError("batch-upload-urls required_headers must be an object")
        if (
            not isinstance(raw.get("expires_in"), int)
            or isinstance(raw.get("expires_in"), bool)
            or int(raw["expires_in"]) <= 0
        ):
            raise RuntimeError("batch-upload-urls expires_in must be a positive integer")
        returned[relative_path] = raw
    missing = sorted(set(requested) - set(returned))
    if missing:
        raise RuntimeError(f"batch-upload-urls response omitted requested files: {missing[:5]}")
    return payload


def decode_batch_complete_response(
    payload: dict[str, Any],
    *,
    project_id: str,
    import_id: str,
    requested_paths: list[str],
) -> dict[str, Any]:
    _require_exact_fields(
        payload,
        {
            "dataset_version_id",
            "embedding_incremental_pipeline",
            "image_ids",
            "import_id",
            "post_upload_pipeline",
            "project_id",
            "status",
            "uploaded_files",
            "version_ref",
            "workspace_id",
        },
        label="batch-complete response",
    )
    _require_identity(payload, project_id=project_id, import_id=import_id)
    if payload.get("status") != "streaming_upload":
        raise RuntimeError("batch-complete status must be streaming_upload")
    if payload.get("uploaded_files") != len(requested_paths):
        raise RuntimeError("batch-complete uploaded_files does not match the requested batch")
    image_ids = payload.get("image_ids")
    if (
        not isinstance(image_ids, list)
        or any(not isinstance(item, str) or not item for item in image_ids)
        or len(set(image_ids)) != len(image_ids)
    ):
        raise RuntimeError("batch-complete image_ids must be unique non-empty strings")
    for key in ("post_upload_pipeline", "embedding_incremental_pipeline"):
        _require_object(payload, key, label="batch-complete response")
    parse_version_ref_identity(payload, label="batch-complete response")
    return payload


def decode_complete_import_response(
    payload: dict[str, Any], *, project_id: str, import_id: str
) -> dict[str, Any]:
    _require_exact_fields(
        payload,
        {
            "dataset_manifest_ref",
            "dataset_version_id",
            "dispatch_mode",
            "import_id",
            "project_id",
            "read_lease",
            "reason",
            "status",
            "version_ref",
            "worker_task_id",
            "workspace_id",
        },
        label="complete-import response",
    )
    _require_identity(payload, project_id=project_id, import_id=import_id)
    if payload.get("status") != "queued":
        raise RuntimeError("complete-import status must be queued")
    _require_object(payload, "dataset_manifest_ref", label="complete-import response")
    _require_object(payload, "read_lease", label="complete-import response")
    parse_version_ref_identity(payload, label="complete-import response")
    if payload.get("reason") != "queued":
        raise RuntimeError("complete-import reason must be queued")
    _require_nonempty_string(payload, "dispatch_mode", label="complete-import response")
    _require_nonempty_string(payload, "worker_task_id", label="complete-import response")
    return payload


def decode_import_job_response(
    payload: dict[str, Any], *, project_id: str, import_id: str
) -> dict[str, Any]:
    _require_exact_fields(
        payload,
        {
            "dataset_validation",
            "dataset_version_id",
            "error",
            "import_id",
            "progress",
            "project_id",
            "status",
            "version_ref",
            "workspace_id",
        },
        label="import-job response",
    )
    _require_identity(payload, project_id=project_id, import_id=import_id)
    status = payload.get("status")
    if status not in IMPORT_STATUSES:
        raise RuntimeError(f"import-job response has unsupported status: {status!r}")
    for key in ("progress", "error"):
        if not isinstance(payload.get(key), dict):
            raise RuntimeError(f"import-job response {key} must be an object")
    dataset_validation = payload.get("dataset_validation")
    if dataset_validation is not None and not isinstance(dataset_validation, dict):
        raise RuntimeError("import-job response dataset_validation must be null or an object")
    parse_version_ref_identity(payload, label="import-job response")
    return payload


def parse_version_ref_identity(
    payload: dict[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    dataset_version_id = _require_nonempty_string(payload, "dataset_version_id", label=label)
    version_ref = _require_object(payload, "version_ref", label=label)
    if "id" in version_ref:
        raise RuntimeError(
            f"{label} version_ref dataset_version_id is the sole identity field; id is invalid"
        )
    referenced_version_id = _require_nonempty_string(
        version_ref,
        "dataset_version_id",
        label=f"{label} version_ref",
    )
    if referenced_version_id != dataset_version_id:
        raise RuntimeError(
            f"{label} version_ref dataset_version_id does not match dataset_version_id"
        )
    return version_ref


def _unique_requested_files(
    requested_files: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for item in requested_files:
        relative_path = str(item.get("relative_path") or "")
        if not relative_path or relative_path in result:
            raise RuntimeError("batch-upload-urls request paths must be unique and non-empty")
        result[relative_path] = item
    return result


def _require_identity(
    payload: dict[str, Any], *, project_id: str, import_id: str | None = None
) -> None:
    if payload.get("project_id") != project_id:
        raise RuntimeError("upload response project_id does not match the request")
    _require_nonempty_string(payload, "workspace_id", label="upload response")
    response_import_id = _require_nonempty_string(payload, "import_id", label="upload response")
    if import_id is not None and response_import_id != import_id:
        raise RuntimeError("upload response import_id does not match the request")


def _require_exact_fields(payload: dict[str, Any], expected: set[str], *, label: str) -> None:
    if set(payload) != expected:
        raise RuntimeError(
            f"{label} fields must be exact: expected={sorted(expected)} actual={sorted(payload)}"
        )


def _require_nonempty_string(payload: dict[str, Any], key: str, *, label: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value or value != value.strip():
        raise RuntimeError(f"{label} {key} must be a canonical non-empty string")
    return value


def _require_object(payload: dict[str, Any], key: str, *, label: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict) or not value:
        raise RuntimeError(f"{label} {key} must be a non-empty object")
    return value
