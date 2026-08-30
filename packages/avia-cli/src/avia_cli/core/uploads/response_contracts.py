from __future__ import annotations

from typing import Any, Literal, cast

from avia_cli.core.uploads.contracts import (
    ANOMALIB_CLASSES,
    require_format_task,
    require_object_prefix_uri,
)
from avia_cli.core.uploads.media_types import require_canonical_media_type

IMPORT_ACTIVE_STATUSES = frozenset({"pending_upload", "uploaded", "queued", "running"})
IMPORT_TERMINAL_STATUSES = frozenset({"succeeded", "failed"})
IMPORT_STATUSES = IMPORT_ACTIVE_STATUSES | IMPORT_TERMINAL_STATUSES
ImportJobType = Literal[
    "yolo_folder",
    "coco_folder",
    "imagenet_folder",
    "anomalib_folder",
]
IMPORT_JOB_TYPES: frozenset[ImportJobType] = frozenset(
    {"yolo_folder", "coco_folder", "imagenet_folder", "anomalib_folder"}
)

_SOURCE_IMPORT_PROGRESS_FIELDS = {
    "classes",
    "file_count",
    "format",
    "image_count",
    "manifest_object_key",
    "phase",
    "source_bucket",
    "source_etag",
    "source_kind",
    "source_size_bytes",
    "source_uri",
    "source_version_id",
    "streamed",
    "task_key",
    "total_bytes",
    "uploaded",
}
_SOURCE_IMPORT_REQUEST_FIELDS = {
    "auto_post_processing",
    "classes",
    "format",
    "source_kind",
    "task_key",
    "uri",
}
_BATCH_UPLOAD_URL_REQUEST_FILE_FIELDS = {
    "relative_path",
    "sha256",
    "size_bytes",
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
    format_name = payload.get("format")
    if format_name == "anomalib":
        if classes != list(ANOMALIB_CLASSES):
            raise RuntimeError(
                "anomalib source-import request classes must be exactly ['good', 'bad']"
            )
    elif format_name != "yolo" and classes:
        raise RuntimeError("coco and imagenet source-import request classes must be empty")
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
    for key in ("uploaded", "image_count", "streamed"):
        value = progress.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"source-import progress {key} must be a non-negative integer")
    if progress["uploaded"] != progress["file_count"]:
        raise RuntimeError("source-import progress uploaded must equal file_count")
    if progress["image_count"] != 0 or progress["streamed"] != 0:
        raise RuntimeError(
            "source-import progress image_count and streamed must be zero before materialization"
        )
    for key in ("source_version_id", "source_bucket", "source_etag"):
        _require_nonempty_string(progress, key, label="source-import progress")
    source_size_bytes = progress.get("source_size_bytes")
    if (
        isinstance(source_size_bytes, bool)
        or not isinstance(source_size_bytes, int)
        or source_size_bytes <= 0
    ):
        raise RuntimeError("source-import progress source_size_bytes must be a positive integer")
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
    requested = validate_batch_upload_urls_request(requested_files)
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
        content_type = require_canonical_media_type(
            raw.get("content_type"),
            label=f"batch-upload-urls content_type for {relative_path}",
        )
        expected_sha = str(expected.get("sha256") or "") or None
        if raw.get("sha256") != expected_sha:
            raise RuntimeError(f"batch-upload-urls sha256 mismatch for {relative_path}")
        object_key = _require_nonempty_string(raw, "object_key", label="batch-upload-urls file")
        upload_url = _require_nonempty_string(raw, "upload_url", label="batch-upload-urls file")
        if object_key in object_keys or upload_url in upload_urls:
            raise RuntimeError("batch-upload-urls remote object identities must be unique")
        object_keys.add(object_key)
        upload_urls.add(upload_url)
        _require_signed_content_type_header(
            raw.get("required_headers"),
            content_type=content_type,
            relative_path=relative_path,
        )
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


def validate_batch_upload_urls_request(
    files: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    if not isinstance(files, list) or not 1 <= len(files) <= 1000:
        raise RuntimeError("batch-upload-urls request files must contain 1 to 1000 items")
    return _unique_requested_files(files)


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
            "import_id",
            "project_id",
            "status",
            "uploaded_files",
            "workspace_id",
        },
        label="batch-complete response",
    )
    _require_identity(payload, project_id=project_id, import_id=import_id)
    if payload.get("status") != "streaming_upload":
        raise RuntimeError("batch-complete status must be streaming_upload")
    if payload.get("uploaded_files") != len(requested_paths):
        raise RuntimeError("batch-complete uploaded_files does not match the requested batch")
    return payload


def decode_complete_import_response(
    payload: dict[str, Any], *, project_id: str, import_id: str
) -> dict[str, Any]:
    _require_exact_fields(
        payload,
        {
            "dataset_manifest_ref",
            "dispatch_mode",
            "import_id",
            "project_id",
            "progress",
            "read_lease",
            "reason",
            "status",
            "worker_task_id",
            "workspace_id",
        },
        label="complete-import response",
    )
    _require_identity(payload, project_id=project_id, import_id=import_id)
    if payload.get("status") != "queued":
        raise RuntimeError("complete-import status must be queued")
    _require_object(payload, "dataset_manifest_ref", label="complete-import response")
    _require_object(payload, "progress", label="complete-import response")
    _require_object(payload, "read_lease", label="complete-import response")
    if payload.get("reason") != "queued":
        raise RuntimeError("complete-import reason must be queued")
    _require_nonempty_string(payload, "dispatch_mode", label="complete-import response")
    _require_nonempty_string(payload, "worker_task_id", label="complete-import response")
    return payload


def decode_import_job_response(
    payload: dict[str, Any], *, project_id: str, import_id: str
) -> dict[str, Any]:
    status = payload.get("status")
    if status not in IMPORT_STATUSES:
        raise RuntimeError(f"import-job response has unsupported status: {status!r}")
    base_fields = {
        "error",
        "import_id",
        "job_type",
        "progress",
        "project_id",
        "status",
        "workspace_id",
    }
    identity_fields = {"dataset_version_id", "version_ref"}
    actual_fields = set(payload)
    expected_fields = (
        base_fields | identity_fields if identity_fields.issubset(actual_fields) else base_fields
    )
    _require_exact_fields(payload, expected_fields, label="import-job response")
    _require_identity(payload, project_id=project_id, import_id=import_id)
    _require_import_job_type(payload.get("job_type"))
    for key in ("progress", "error"):
        if not isinstance(payload.get(key), dict):
            raise RuntimeError(f"import-job response {key} must be an object")
    validate_version_ref_phase(
        payload,
        status=str(status),
        label="import-job response",
    )
    return payload


def _require_import_job_type(value: object) -> ImportJobType:
    if not isinstance(value, str) or value not in IMPORT_JOB_TYPES:
        raise RuntimeError(f"import-job response has unsupported job_type: {value!r}")
    return cast(ImportJobType, value)


def validate_version_ref_phase(
    payload: dict[str, Any],
    *,
    status: str,
    label: str,
) -> dict[str, Any] | None:
    has_dataset_version_id = "dataset_version_id" in payload
    has_version_ref = "version_ref" in payload
    if status == "succeeded":
        if not has_dataset_version_id or not has_version_ref:
            raise RuntimeError(
                f"{label} succeeded status requires dataset_version_id and version_ref"
            )
        return parse_version_ref_identity(payload, label=label)
    if has_dataset_version_id != has_version_ref:
        raise RuntimeError(f"{label} pre-published identity fields must both be absent or null")
    if has_dataset_version_id and (
        payload.get("dataset_version_id") is not None or payload.get("version_ref") is not None
    ):
        raise RuntimeError(
            f"{label} must not expose dataset_version_id or version_ref before succeeded"
        )
    return None


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
        if not isinstance(item, dict):
            raise RuntimeError("batch-upload-urls request file must be an object")
        _require_exact_fields(
            item,
            _BATCH_UPLOAD_URL_REQUEST_FILE_FIELDS,
            label="batch-upload-urls request file",
        )
        relative_path = _require_nonempty_string(
            item,
            "relative_path",
            label="batch-upload-urls request file",
        )
        if relative_path in result:
            raise RuntimeError("batch-upload-urls request paths must be unique and non-empty")
        size_bytes = item.get("size_bytes")
        if isinstance(size_bytes, bool) or not isinstance(size_bytes, int) or size_bytes < 0:
            raise RuntimeError(
                f"batch-upload-urls request size_bytes is invalid for {relative_path}"
            )
        sha256 = item.get("sha256")
        if (
            not isinstance(sha256, str)
            or len(sha256) != 64
            or any(character not in "0123456789abcdef" for character in sha256)
        ):
            raise RuntimeError(f"batch-upload-urls request sha256 is invalid for {relative_path}")
        result[relative_path] = item
    return result


def _require_signed_content_type_header(
    value: object,
    *,
    content_type: str,
    relative_path: str,
) -> None:
    if not isinstance(value, dict):
        raise RuntimeError("batch-upload-urls required_headers must be an object")
    normalized: dict[str, str] = {}
    for raw_name, raw_value in value.items():
        if (
            not isinstance(raw_name, str)
            or not raw_name
            or raw_name != raw_name.strip()
            or not isinstance(raw_value, str)
            or not raw_value
            or raw_value != raw_value.strip()
        ):
            raise RuntimeError(
                f"batch-upload-urls required_headers are invalid for {relative_path}"
            )
        name = raw_name.lower()
        if name in normalized:
            raise RuntimeError(f"batch-upload-urls required_headers duplicate {raw_name}")
        normalized[name] = raw_value
    if normalized != {"content-type": content_type}:
        raise RuntimeError(
            "batch-upload-urls required Content-Type header does not match "
            f"server content_type for {relative_path}"
        )


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
