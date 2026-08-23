from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from avia_cli.context import api_from_args
from avia_cli.core.api_base import canonical_api_base
from avia_cli.core.uploads.api import _complete_dataset_file_batch
from avia_cli.core.uploads.dataset import create_source_import
from avia_cli.core.uploads.refs import attach_upload_refs
from avia_cli.core.uploads.response_contracts import (
    IMPORT_JOB_TYPES,
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
            "uploaded": 3,
            "image_count": 0,
            "streamed": 0,
            "manifest_object_key": "workspaces/ws_123/imports/imp_123/manifest.json",
            "source_version_id": "version-123",
            "source_bucket": "datasets",
            "source_etag": "etag-123",
            "source_size_bytes": 456,
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


def test_source_import_rejects_class_override_for_imagenet() -> None:
    request_payload, _response = _source_import_contract(auto_post_processing=False)
    request_payload.update({"format": "imagenet", "task_key": "classify"})

    with pytest.raises(RuntimeError, match=r"coco and imagenet.*classes must be empty"):
        validate_source_import_request(request_payload)


def test_source_import_requires_exact_anomalib_binary_taxonomy() -> None:
    request_payload, _response = _source_import_contract(auto_post_processing=False)
    request_payload.update(
        {
            "format": "anomalib",
            "task_key": "ad",
            "classes": ["good", "bad"],
        }
    )

    validate_source_import_request(request_payload)
    for invalid in ([], ["bad", "good"], ["good"], ["good", "bad", "other"]):
        with pytest.raises(RuntimeError, match="exactly"):
            validate_source_import_request({**request_payload, "classes": invalid})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_version_id", ""),
        ("source_bucket", ""),
        ("source_etag", ""),
        ("source_size_bytes", 0),
    ],
)
def test_source_import_response_requires_immutable_manifest_identity(
    field: str,
    value: object,
) -> None:
    request_payload, response = _source_import_contract(auto_post_processing=False)
    progress = dict(response["progress"])
    progress[field] = value

    with pytest.raises(RuntimeError, match=field):
        decode_source_import_response(
            {**response, "progress": progress},
            project_id=PROJECT_ID,
            request_payload=request_payload,
        )


def test_source_import_progress_matches_yolo_control_plane_exact_shape() -> None:
    request_payload, response = _source_import_contract(auto_post_processing=False)
    progress = response["progress"]

    assert isinstance(progress, dict)
    assert set(progress) == {
        "source_kind",
        "source_uri",
        "format",
        "task_key",
        "classes",
        "file_count",
        "total_bytes",
        "uploaded",
        "image_count",
        "streamed",
        "manifest_object_key",
        "source_version_id",
        "source_bucket",
        "source_etag",
        "source_size_bytes",
        "phase",
    }
    assert (
        decode_source_import_response(
            response,
            project_id=PROJECT_ID,
            request_payload=request_payload,
        )
        is response
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("uploaded", 2, "uploaded must equal file_count"),
        ("image_count", 1, "image_count and streamed must be zero"),
        ("streamed", 1, "image_count and streamed must be zero"),
    ],
)
def test_source_import_progress_rejects_pre_materialization_counter_drift(
    field: str,
    value: int,
    message: str,
) -> None:
    request_payload, response = _source_import_contract(auto_post_processing=False)
    progress = dict(response["progress"])
    progress[field] = value

    with pytest.raises(RuntimeError, match=message):
        decode_source_import_response(
            {**response, "progress": progress},
            project_id=PROJECT_ID,
            request_payload=request_payload,
        )


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
            "sha256": "a" * 64,
        }
    ]
    item = {
        **requested[0],
        "content_type": "image/png",
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


def test_batch_upload_decoder_treats_content_type_as_server_owned() -> None:
    requested = [
        {
            "relative_path": "metadata/data.yaml",
            "size_bytes": 12,
            "sha256": "b" * 64,
        }
    ]
    item = {
        **requested[0],
        "content_type": "application/yaml",
        "object_key": "objects/data.yaml",
        "upload_url": "https://storage.example/data?signature=secret",
        "required_headers": {"Content-Type": "application/yaml"},
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
    with pytest.raises(RuntimeError, match="Content-Type"):
        decode_batch_upload_urls_response(
            {
                **payload,
                "files": [
                    {
                        **item,
                        "required_headers": {"Content-Type": "text/yaml"},
                    }
                ],
            },
            project_id=PROJECT_ID,
            import_id=IMPORT_ID,
            requested_files=requested,
        )
    with pytest.raises(RuntimeError, match="request file fields must be exact"):
        decode_batch_upload_urls_response(
            payload,
            project_id=PROJECT_ID,
            import_id=IMPORT_ID,
            requested_files=[{**requested[0], "content_type": "text/yaml"}],
        )


@pytest.mark.parametrize(
    "content_type",
    [
        "",
        "/png",
        "image/",
        "image//png",
        " image/png",
        "image/png ",
        "image /png",
        "image/\npng",
        "image/png; charset=utf-8",
        "IMAGE/PNG",
        "image/π",
    ],
)
def test_batch_upload_decoder_rejects_noncanonical_media_types(
    content_type: str,
) -> None:
    requested = [
        {
            "relative_path": "images/train/a.png",
            "size_bytes": 12,
            "sha256": "a" * 64,
        }
    ]
    item = {
        **requested[0],
        "content_type": content_type,
        "object_key": "objects/a.png",
        "upload_url": "https://storage.example/a?signature=secret",
        "required_headers": {"Content-Type": content_type},
        "expires_in": 900,
    }

    with pytest.raises(RuntimeError, match="content_type"):
        decode_batch_upload_urls_response(
            {
                "workspace_id": "ws_123",
                "project_id": PROJECT_ID,
                "import_id": IMPORT_ID,
                "files": [item],
            },
            project_id=PROJECT_ID,
            import_id=IMPORT_ID,
            requested_files=requested,
        )


@pytest.mark.parametrize("content_type", ["image//png", "image/png ", "image/\x00png"])
def test_batch_complete_manifest_reuses_canonical_media_type_contract(
    content_type: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "avia_cli.core.uploads.api._request_json",
        lambda **_kwargs: pytest.fail("invalid manifest media type reached HTTP"),
    )

    with pytest.raises(RuntimeError, match="batch-complete content_type"):
        _complete_dataset_file_batch(
            api="https://avia.example/api/v1",
            token="token",
            project_id=PROJECT_ID,
            import_id=IMPORT_ID,
            files=[
                {
                    "relative_path": "images/train/a.png",
                    "object_key": "objects/a.png",
                    "size_bytes": 12,
                    "content_type": content_type,
                    "sha256": "a" * 64,
                    "width": 16,
                    "height": 12,
                }
            ],
        )


def test_batch_complete_decoder_requires_proof_of_exact_accepted_batch() -> None:
    payload = {
        "workspace_id": "ws_123",
        "project_id": PROJECT_ID,
        "import_id": IMPORT_ID,
        "status": "streaming_upload",
        "uploaded_files": 1,
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
    with pytest.raises(RuntimeError, match="fields must be exact"):
        decode_batch_complete_response(
            {**payload, "dataset_version_id": "dv_123"},
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
        "progress": {"phase": "queued"},
    }
    assert decode_complete_import_response(complete, project_id=PROJECT_ID, import_id=IMPORT_ID)
    job = {
        "workspace_id": "ws_123",
        "project_id": PROJECT_ID,
        "import_id": IMPORT_ID,
        "job_type": "yolo_folder",
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


@pytest.mark.parametrize("job_type", sorted(IMPORT_JOB_TYPES))
def test_import_job_requires_one_current_folder_job_type(job_type: str) -> None:
    job = {
        "workspace_id": "ws_123",
        "project_id": PROJECT_ID,
        "import_id": IMPORT_ID,
        "job_type": job_type,
        "status": "running",
        "progress": {"phase": "running"},
        "error": {},
        "dataset_validation": None,
    }

    assert decode_import_job_response(job, project_id=PROJECT_ID, import_id=IMPORT_ID) is job


@pytest.mark.parametrize(
    "job_type",
    [None, "", " yolo_folder", "dataset.import.yolo", "yolo", "unknown_folder"],
)
def test_import_job_rejects_missing_legacy_or_unknown_job_type(job_type: object) -> None:
    job = {
        "workspace_id": "ws_123",
        "project_id": PROJECT_ID,
        "import_id": IMPORT_ID,
        "job_type": job_type,
        "status": "running",
        "progress": {"phase": "running"},
        "error": {},
        "dataset_validation": None,
    }

    if job_type is None:
        job.pop("job_type")
        message = "fields must be exact"
    else:
        message = "unsupported job_type"
    with pytest.raises(RuntimeError, match=message):
        decode_import_job_response(job, project_id=PROJECT_ID, import_id=IMPORT_ID)


@pytest.mark.parametrize(
    "identity",
    [
        {"dataset_version_id": None, "version_ref": None},
        {"dataset_version_id": "dv_123", "version_ref": None},
        {
            "dataset_version_id": "dv_123",
            "version_ref": {"dataset_version_id": "dv_123"},
        },
    ],
)
def test_complete_import_rejects_prepublication_version_identity(
    identity: dict[str, object],
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
        **identity,
    }

    with pytest.raises(RuntimeError, match="fields must be exact"):
        decode_complete_import_response(payload, project_id=PROJECT_ID, import_id=IMPORT_ID)


@pytest.mark.parametrize("status", sorted(IMPORT_STATUSES - {"succeeded"}))
@pytest.mark.parametrize("identity_mode", ["absent", "null"])
def test_prepublication_import_job_accepts_only_absent_or_null_version_identity(
    status: str,
    identity_mode: str,
) -> None:
    job = {
        "workspace_id": "ws_123",
        "project_id": PROJECT_ID,
        "import_id": IMPORT_ID,
        "job_type": "yolo_folder",
        "status": status,
        "progress": {"phase": status},
        "error": {},
        "dataset_validation": None,
    }
    if identity_mode == "null":
        job.update({"dataset_version_id": None, "version_ref": None})

    assert decode_import_job_response(job, project_id=PROJECT_ID, import_id=IMPORT_ID) is job


@pytest.mark.parametrize("status", sorted(IMPORT_STATUSES - {"succeeded"}))
def test_prepublication_import_job_rejects_product_version_identity(status: str) -> None:
    job = {
        "workspace_id": "ws_123",
        "project_id": PROJECT_ID,
        "import_id": IMPORT_ID,
        "job_type": "yolo_folder",
        "status": status,
        "progress": {"phase": status},
        "error": {},
        "dataset_validation": None,
        "dataset_version_id": "dv_123",
        "version_ref": {"dataset_version_id": "dv_123"},
    }

    with pytest.raises(RuntimeError, match="must not expose"):
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
        "job_type": "yolo_folder",
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
        "job_type": "yolo_folder",
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
