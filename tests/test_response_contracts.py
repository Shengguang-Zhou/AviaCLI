from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from avia_cli.context import api_from_args
from avia_cli.core.api_base import canonical_api_base
from avia_cli.core.uploads.dataset import create_source_import
from avia_cli.core.uploads.refs import attach_upload_refs
from avia_cli.core.uploads.response_contracts import (
    IMPORT_STATUSES,
    decode_batch_complete_response,
    decode_batch_upload_urls_response,
    decode_complete_import_response,
    decode_dataset_session_response,
    decode_import_job_response,
    decode_source_import_response,
    validate_source_import_request,
)

PROJECT_ID = "proj_123456789abc"
IMPORT_ID = "imp_123"
_IMPORT_COMPLETE_CONTRACT = json.loads(
    Path("tests/fixtures/import_complete_queued_v1.json").read_text(encoding="utf-8")
)


def _ref_payload() -> tuple[dict[str, object], dict[str, object]]:
    ref = {"id": "dm_123", "storage": {"kind": "minio", "manifest_path": "m.json"}}
    lease = {"id": "lease_123", "dataset_manifest_ref_id": "dm_123"}
    return ref, lease


def _source_import_contract(
    *, auto_post_processing: bool
) -> tuple[dict[str, object], dict[str, object]]:
    request_payload: dict[str, object] = {
        "source_kind": "object_prefix",
        "uri": "datasets/coco8/",
        "format": "yolo",
        "task_key": "detect",
        "classes": ["person"],
        "auto_post_processing": auto_post_processing,
    }
    response: dict[str, object] = {
        "workspace_id": "ws_123",
        "project_id": PROJECT_ID,
        "import_id": IMPORT_ID,
        "status": "queued" if auto_post_processing else "uploaded",
        "object_key": "workspaces/ws_123/imports/imp_123/manifest.json",
        "progress": {
            "source_kind": "object_prefix",
            "source_uri": "datasets/coco8/",
            "format": "yolo",
            "task_key": "detect",
            "classes": ["person"],
            "file_count": 3,
            "total_bytes": 123,
            "manifest_object_key": "workspaces/ws_123/imports/imp_123/manifest.json",
            "source_owned": False,
            "all_referenced_existing": False,
            "phase": "uploaded",
        },
    }
    if auto_post_processing:
        response.update(
            {
                "reason": "queued",
                "dispatch_mode": "celery",
                "worker_task_id": "task_123",
            }
        )
    return request_payload, response


def test_source_import_rejects_class_override_for_non_yolo_format() -> None:
    request_payload, _response = _source_import_contract(auto_post_processing=False)
    request_payload.update({"format": "imagenet", "task_key": "classify"})

    with pytest.raises(RuntimeError, match="classes are only valid for yolo"):
        validate_source_import_request(request_payload)


@pytest.mark.parametrize(
    "value",
    [
        "avia.example/api/v1",
        "ftp://avia.example/api/v1",
        "https://user@avia.example/api/v1",
        "https://avia.example/api/v1?tenant=x",
        "https://avia.example/api/v1#fragment",
        "https://avia.example/api/v1/",
        " https://avia.example/api/v1",
        "https://avia.example:443/api/v1",
    ],
)
def test_api_base_rejects_every_noncanonical_form(value: str) -> None:
    with pytest.raises(ValueError):
        canonical_api_base(value)


def test_api_from_args_uses_the_same_canonical_contract() -> None:
    with pytest.raises(SystemExit, match="not canonical"):
        api_from_args(SimpleNamespace(api="https://avia.example/api/v1/"))


@pytest.mark.parametrize("value", [" https://avia.example/api/v1", "https://avia.example/api/v1 "])
def test_api_from_args_does_not_silently_trim_cli_whitespace(value: str) -> None:
    with pytest.raises(SystemExit, match="must not contain whitespace"):
        api_from_args(SimpleNamespace(api=value))


def test_api_from_args_does_not_silently_trim_environment_whitespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AVIA_API_BASE", " https://avia.example/api/v1")

    with pytest.raises(SystemExit, match="must not contain whitespace"):
        api_from_args(SimpleNamespace(api=""))


def test_dataset_session_decoder_accepts_only_exact_backend_contract() -> None:
    ref, lease = _ref_payload()
    payload = {
        "workspace_id": "ws_123",
        "project_id": PROJECT_ID,
        "import_id": IMPORT_ID,
        "status": "pending_upload",
        "object_key": "workspace/imports/manifest.json",
        "dataset_manifest_ref": ref,
        "read_lease": lease,
    }

    assert decode_dataset_session_response(payload, project_id=PROJECT_ID) is payload
    with pytest.raises(RuntimeError, match="fields must be exact"):
        decode_dataset_session_response({**payload, "legacy_id": "x"}, project_id=PROJECT_ID)


@pytest.mark.parametrize("auto_post_processing", [False, True])
def test_source_import_decoder_accepts_only_exact_auto_processing_envelope(
    auto_post_processing: bool,
) -> None:
    request_payload, response = _source_import_contract(auto_post_processing=auto_post_processing)

    assert (
        decode_source_import_response(
            response,
            project_id=PROJECT_ID,
            request_payload=request_payload,
        )
        is response
    )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda response: response.clear(),
        lambda response: response.update({"unknown": True}),
        lambda response: response.update({"status": "succeeded"}),
        lambda response: response.update({"project_id": "proj_other"}),
        lambda response: response["progress"].update({"task_key": "pose"}),
        lambda response: response["progress"].update({"source_uri": "datasets/other/"}),
    ],
)
def test_source_import_decoder_rejects_empty_unknown_status_and_identity_drift(
    mutation,
) -> None:
    request_payload, response = _source_import_contract(auto_post_processing=False)
    mutation(response)

    with pytest.raises(RuntimeError):
        decode_source_import_response(
            response,
            project_id=PROJECT_ID,
            request_payload=request_payload,
        )


def test_create_source_import_rejects_empty_2xx_object(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_payload, _response = _source_import_contract(auto_post_processing=False)
    monkeypatch.setattr("avia_cli.core.uploads.dataset._post_json", lambda **_kwargs: {})

    with pytest.raises(RuntimeError, match="source-import response fields must be exact"):
        create_source_import(
            api="https://avia.example/api/v1",
            token="token",
            project_id=PROJECT_ID,
            payload=request_payload,
        )


def test_create_source_import_rejects_noncanonical_uri_before_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request_payload, _response = _source_import_contract(auto_post_processing=False)
    request_payload["uri"] = "s3://bucket/datasets/coco8"
    monkeypatch.setattr(
        "avia_cli.core.uploads.dataset._post_json",
        lambda **_kwargs: pytest.fail("invalid source-import request reached HTTP"),
    )

    with pytest.raises(RuntimeError, match="canonical bare object path"):
        create_source_import(
            api="https://avia.example/api/v1",
            token="token",
            project_id=PROJECT_ID,
            payload=request_payload,
        )


def test_batch_upload_decoder_rejects_duplicate_extra_missing_and_identity_drift() -> None:
    requested = [
        {
            "relative_path": "images/train/a.png",
            "size_bytes": 12,
            "content_type": "image/png",
            "sha256": "a" * 64,
        }
    ]
    item = {
        **requested[0],
        "object_key": "objects/a.png",
        "upload_url": "https://storage.example/a?signature=secret",
        "required_headers": {"Content-Type": "image/png"},
        "expires_in": 900,
    }
    payload = {
        "workspace_id": "ws_123",
        "project_id": PROJECT_ID,
        "import_id": IMPORT_ID,
        "files": [item],
    }
    assert (
        decode_batch_upload_urls_response(
            payload,
            project_id=PROJECT_ID,
            import_id=IMPORT_ID,
            requested_files=requested,
        )
        is payload
    )
    for files in ([item, item], [], [{**item, "relative_path": "extra.png"}]):
        with pytest.raises(RuntimeError):
            decode_batch_upload_urls_response(
                {**payload, "files": files},
                project_id=PROJECT_ID,
                import_id=IMPORT_ID,
                requested_files=requested,
            )


def test_batch_complete_decoder_requires_proof_of_exact_accepted_batch() -> None:
    payload = {
        "workspace_id": "ws_123",
        "project_id": PROJECT_ID,
        "import_id": IMPORT_ID,
        "status": "streaming_upload",
        "uploaded_files": 1,
        "image_ids": ["img_123"],
        "dataset_version_id": "dv_123",
        "version_ref": {"dataset_version_id": "dv_123"},
        "post_upload_pipeline": {"status": "queued"},
        "embedding_incremental_pipeline": {"status": "skipped"},
    }
    assert (
        decode_batch_complete_response(
            payload,
            project_id=PROJECT_ID,
            import_id=IMPORT_ID,
            requested_paths=["images/train/a.png"],
        )
        is payload
    )
    with pytest.raises(RuntimeError, match="uploaded_files"):
        decode_batch_complete_response(
            {**payload, "uploaded_files": 0},
            project_id=PROJECT_ID,
            import_id=IMPORT_ID,
            requested_paths=["images/train/a.png"],
        )
    with pytest.raises(RuntimeError, match="version_ref dataset_version_id"):
        decode_batch_complete_response(
            {**payload, "version_ref": {"id": "dv_123"}},
            project_id=PROJECT_ID,
            import_id=IMPORT_ID,
            requested_paths=["images/train/a.png"],
        )


def test_complete_and_poll_decoders_reject_historical_status_aliases() -> None:
    ref, lease = _ref_payload()
    complete = {
        "workspace_id": "ws_123",
        "project_id": PROJECT_ID,
        "import_id": IMPORT_ID,
        "status": "queued",
        "dataset_manifest_ref": ref,
        "read_lease": lease,
        "reason": "queued",
        "dispatch_mode": "celery",
        "worker_task_id": "task_123",
        "dataset_version_id": "dv_123",
        "version_ref": {"dataset_version_id": "dv_123"},
    }
    assert decode_complete_import_response(complete, project_id=PROJECT_ID, import_id=IMPORT_ID)
    job = {
        "workspace_id": "ws_123",
        "project_id": PROJECT_ID,
        "import_id": IMPORT_ID,
        "status": "succeeded",
        "progress": {"phase": "succeeded"},
        "error": {},
        "dataset_validation": None,
        "dataset_version_id": "dv_123",
        "version_ref": {"dataset_version_id": "dv_123"},
    }
    assert decode_import_job_response(job, project_id=PROJECT_ID, import_id=IMPORT_ID)
    for alias in ("success", "completed", "done", "error", "cancelled"):
        with pytest.raises(RuntimeError, match="unsupported status"):
            decode_import_job_response(
                {**job, "status": alias}, project_id=PROJECT_ID, import_id=IMPORT_ID
            )


@pytest.mark.parametrize(
    "version_ref",
    [
        {"id": "dv_123"},
        {"dataset_version_id": ""},
        {"dataset_version_id": " dv_123"},
        {"dataset_version_id": "dv_other"},
        {"dataset_version_id": "dv_123", "id": "dv_123"},
    ],
)
def test_complete_import_rejects_noncanonical_version_reference_identity(
    version_ref: dict[str, object],
) -> None:
    ref, lease = _ref_payload()
    payload = {
        "workspace_id": "ws_123",
        "project_id": PROJECT_ID,
        "import_id": IMPORT_ID,
        "status": "queued",
        "dataset_manifest_ref": ref,
        "read_lease": lease,
        "reason": "queued",
        "dispatch_mode": "celery",
        "worker_task_id": "task_123",
        "dataset_version_id": "dv_123",
        "version_ref": version_ref,
    }

    with pytest.raises(RuntimeError, match="version_ref dataset_version_id"):
        decode_complete_import_response(payload, project_id=PROJECT_ID, import_id=IMPORT_ID)


@pytest.mark.parametrize("status", sorted(IMPORT_STATUSES))
def test_every_import_job_status_rejects_historical_version_reference(status: str) -> None:
    job = {
        "workspace_id": "ws_123",
        "project_id": PROJECT_ID,
        "import_id": IMPORT_ID,
        "status": status,
        "progress": {"phase": status},
        "error": {},
        "dataset_validation": None,
        "dataset_version_id": "dv_123",
        "version_ref": {"id": "dv_123"},
    }

    with pytest.raises(RuntimeError, match="version_ref dataset_version_id"):
        decode_import_job_response(job, project_id=PROJECT_ID, import_id=IMPORT_ID)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dataset_version_id", ""),
        ("version_ref", None),
        ("dataset_validation", "not-an-object"),
    ],
)
def test_succeeded_import_job_requires_usable_result_references(field: str, value: object) -> None:
    job = {
        "workspace_id": "ws_123",
        "project_id": PROJECT_ID,
        "import_id": IMPORT_ID,
        "status": "succeeded",
        "progress": {"phase": "succeeded"},
        "error": {},
        "dataset_validation": None,
        "dataset_version_id": "dv_123",
        "version_ref": {"dataset_version_id": "dv_123"},
    }
    job[field] = value

    with pytest.raises(RuntimeError, match=field):
        decode_import_job_response(job, project_id=PROJECT_ID, import_id=IMPORT_ID)


def test_succeeded_import_job_rejects_conflicting_version_reference_identity() -> None:
    job = {
        "workspace_id": "ws_123",
        "project_id": PROJECT_ID,
        "import_id": IMPORT_ID,
        "status": "succeeded",
        "progress": {"phase": "succeeded"},
        "error": {},
        "dataset_validation": None,
        "dataset_version_id": "dv_123",
        "version_ref": {"dataset_version_id": "dv_other"},
    }

    with pytest.raises(RuntimeError, match="version_ref dataset_version_id"):
        decode_import_job_response(job, project_id=PROJECT_ID, import_id=IMPORT_ID)


def test_complete_decoder_accepts_the_shared_server_queued_contract() -> None:
    contract = _IMPORT_COMPLETE_CONTRACT
    payload = dict(contract["example"])

    assert set(payload) == set(contract["fields"])
    assert (
        decode_complete_import_response(
            payload,
            project_id=str(payload["project_id"]),
            import_id=str(payload["import_id"]),
        )
        is payload
    )
    for field in contract["nonempty_strings"]:
        if field in {"project_id", "import_id", "workspace_id"}:
            continue
        with pytest.raises(RuntimeError, match=field):
            decode_complete_import_response(
                {**payload, field: ""},
                project_id=str(payload["project_id"]),
                import_id=str(payload["import_id"]),
            )
    for field in contract["object_fields"]:
        with pytest.raises(RuntimeError, match=field):
            decode_complete_import_response(
                {**payload, field: {}},
                project_id=str(payload["project_id"]),
                import_id=str(payload["import_id"]),
            )


def test_attach_upload_refs_rejects_conflicting_identity_instead_of_first_wins() -> None:
    result = {
        "complete": {"dataset_manifest_ref": {"id": "dm_first"}},
        "job": {"dataset_manifest_ref": {"id": "dm_second"}},
    }

    with pytest.raises(RuntimeError, match="conflicting dataset_manifest_ref"):
        attach_upload_refs(result)


def test_concurrent_upload_error_is_machine_readable() -> None:
    from avia_cli.core.errors import ConcurrentUploadError

    error = ConcurrentUploadError(
        [
            ("file-put", ["a.png"], RuntimeError("put failed")),
            ("batch-complete", ["b.txt"], ValueError("complete failed")),
        ]
    )

    payload = json.loads(str(error))
    assert payload["code"] == "concurrent_upload_failed"
    assert payload["failure_count"] == 2
    assert payload["failures"][1]["targets"] == ["b.txt"]
