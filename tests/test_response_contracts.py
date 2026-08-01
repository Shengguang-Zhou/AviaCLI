from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from avia_cli.context import api_from_args
from avia_cli.core.api_base import canonical_api_base
from avia_cli.core.uploads.api import _complete_dataset_file_batch
from avia_cli.core.uploads.dataset import create_source_import
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


def _session_request_payload(
    *,
    format_name: str = "yolo",
    task_key: str = "detect",
    file_count: int = 3,
    total_bytes: int = 123,
) -> dict[str, object]:
    return {
        "idempotency_key": "5d74e1c1-f1e4-4b4b-9b42-cae872f71c4a",
        "format": format_name,
        "root_name": "dataset",
        "task_key": task_key,
        "classes": ["person"],
        "file_count": file_count,
        "total_bytes": total_bytes,
    }


def _ref_payload(
    *,
    import_id: str = IMPORT_ID,
    object_key: str | None = None,
    format_name: str = "yolo",
    item_count: int = 3,
    byte_count: int = 123,
) -> tuple[dict[str, object], dict[str, object]]:
    manifest_path = (
        object_key or f"project_assets/ws_123/scope_123/imports/{import_id}/manifest.json"
    )
    ref_id = f"dm_{import_id}"
    ref = {
        "id": ref_id,
        "format": format_name,
        "item_count": item_count,
        "byte_count": byte_count,
        "storage": {
            "kind": "minio",
            "manifest_path": manifest_path,
            "path_prefix": manifest_path.rsplit("/", 1)[0],
            "lakefs_repo": None,
            "lakefs_commit": None,
            "dataset_version_id": None,
        },
    }
    lease = {
        "id": f"lease_{import_id}",
        "scope": "read",
        "access": "object_ref",
        "dataset_manifest_ref_id": ref_id,
    }
    return ref, lease


def _session_response(
    *,
    project_id: str = PROJECT_ID,
    import_id: str = IMPORT_ID,
    workspace_id: str = "ws_123",
    request_payload: dict[str, object] | None = None,
    object_key: str | None = None,
) -> dict[str, object]:
    request = request_payload or _session_request_payload()
    manifest_path = (
        object_key or f"project_assets/{workspace_id}/scope_123/imports/{import_id}/manifest.json"
    )
    ref, lease = _ref_payload(
        import_id=import_id,
        object_key=manifest_path,
        format_name=str(request["format"]),
        item_count=int(request["file_count"]),
        byte_count=int(request["total_bytes"]),
    )
    return {
        "workspace_id": workspace_id,
        "project_id": project_id,
        "import_id": import_id,
        "status": "pending_upload",
        "object_key": manifest_path,
        "dataset_manifest_ref": ref,
        "read_lease": lease,
    }


def _materialized_version_ref(
    dataset_version_id: str = "dsv_123",
    project_scope_id: str = "scope_123",
) -> dict[str, object]:
    path_prefix = f"dataset-manifests/{project_scope_id}/{dataset_version_id}"
    return {
        "dataset_version_id": dataset_version_id,
        "storage_kind": "minio_lakefs",
        "lakefs_repo": "avia-datasets",
        "lakefs_commit": "commit-123",
        "lakefs_tag": dataset_version_id,
        "path_prefix": path_prefix,
        "manifest_path": f"{path_prefix}/manifest.json",
        "content_digest": f"sha256:{'a' * 64}",
        "item_count": 1,
        "byte_count": 12,
    }


def _decode_batch_urls(
    payload: dict[str, object],
    *,
    requested_files: list[dict[str, object]],
) -> dict[str, object]:
    return decode_batch_upload_urls_response(
        payload,
        project_id=PROJECT_ID,
        import_id=IMPORT_ID,
        requested_files=requested_files,
        request_payload=_session_request_payload(),
        session_response=_session_response(),
    )


def _decode_job(payload: dict[str, object]) -> dict[str, object]:
    return decode_import_job_response(
        payload,
        project_id=PROJECT_ID,
        import_id=IMPORT_ID,
        request_payload=_session_request_payload(),
        session_response=_session_response(),
    )


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
        "object_key": "project_assets/ws_123/scope_123/imports/imp_123/manifest.json",
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
            "manifest_object_key": (
                "project_assets/ws_123/scope_123/imports/imp_123/manifest.json"
            ),
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

    with pytest.raises(RuntimeError, match=r"classes must be empty for imagenet"):
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
    ("format_name", "task_key", "classes"),
    [
        ("yolo", "detect", []),
        ("yolo", "obb", ["aircraft"]),
        ("coco", "segment", []),
        ("imagenet", "classify", []),
        ("anomalib", "ad", ["good", "bad"]),
    ],
)
def test_source_import_uses_one_exact_object_prefix_class_matrix(
    format_name: str,
    task_key: str,
    classes: list[str],
) -> None:
    request_payload, _response = _source_import_contract(auto_post_processing=False)
    request_payload.update({"format": format_name, "task_key": task_key, "classes": classes})

    validate_source_import_request(request_payload)


@pytest.mark.parametrize(
    ("format_name", "task_key", "classes"),
    [
        ("coco", "detect", ["person"]),
        ("imagenet", "classify", ["aircraft"]),
        ("anomalib", "ad", []),
        ("anomalib", "ad", ["bad", "good"]),
    ],
)
def test_source_import_rejects_noncanonical_object_prefix_class_matrix(
    format_name: str,
    task_key: str,
    classes: list[str],
) -> None:
    request_payload, _response = _source_import_contract(auto_post_processing=False)
    request_payload.update({"format": format_name, "task_key": task_key, "classes": classes})

    with pytest.raises(RuntimeError):
        validate_source_import_request(request_payload)


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


@pytest.mark.parametrize(
    "source_version_id",
    [
        " \n版本-α\t ",
        " null ",
        "v" * 1024,
        "界" * 341 + "a",
    ],
)
def test_source_import_preserves_opaque_s3_version_id(
    source_version_id: str,
) -> None:
    request_payload, response = _source_import_contract(auto_post_processing=False)
    progress = dict(response["progress"])
    progress["source_version_id"] = source_version_id
    payload = {**response, "progress": progress}

    assert (
        decode_source_import_response(
            payload,
            project_id=PROJECT_ID,
            request_payload=request_payload,
        )
        is payload
    )
    assert progress["source_version_id"] == source_version_id


@pytest.mark.parametrize(
    "source_version_id",
    [
        "",
        "null",
        "v" * 1025,
        None,
        1,
        b"version",
        "\ud800",
    ],
)
def test_source_import_rejects_invalid_s3_version_id(
    source_version_id: object,
) -> None:
    request_payload, response = _source_import_contract(auto_post_processing=False)
    progress = dict(response["progress"])
    progress["source_version_id"] = source_version_id

    with pytest.raises(RuntimeError, match="source_version_id"):
        decode_source_import_response(
            {**response, "progress": progress},
            project_id=PROJECT_ID,
            request_payload=request_payload,
        )


@pytest.mark.parametrize("field", ["source_bucket", "source_etag"])
def test_s3_version_id_opaque_rules_do_not_weaken_other_source_identity_fields(
    field: str,
) -> None:
    request_payload, response = _source_import_contract(auto_post_processing=False)
    progress = dict(response["progress"])
    progress["source_version_id"] = " \n版本\t "
    progress[field] = " padded "

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
    object_key = "project_assets/ws_123/scope_123/imports/imp_123/manifest.json"
    payload = {
        "workspace_id": "ws_123",
        "project_id": PROJECT_ID,
        "import_id": IMPORT_ID,
        "status": "pending_upload",
        "object_key": object_key,
        "dataset_manifest_ref": ref,
        "read_lease": lease,
    }

    assert (
        decode_dataset_session_response(
            payload,
            project_id=PROJECT_ID,
            request_payload=_session_request_payload(),
        )
        is payload
    )
    with pytest.raises(RuntimeError, match="fields must be exact"):
        decode_dataset_session_response(
            {**payload, "legacy_id": "x"},
            project_id=PROJECT_ID,
            request_payload=_session_request_payload(),
        )


def test_dataset_session_rejects_malformed_or_foreign_manifest_identities() -> None:
    ref, lease = _ref_payload()
    foreign_workspace_ref, _foreign_workspace_lease = _ref_payload(
        object_key=("project_assets/ws_foreign/scope_123/imports/imp_123/manifest.json")
    )
    payload = {
        "workspace_id": "ws_123",
        "project_id": PROJECT_ID,
        "import_id": IMPORT_ID,
        "status": "pending_upload",
        "object_key": "project_assets/ws_123/scope_123/imports/imp_123/manifest.json",
        "dataset_manifest_ref": ref,
        "read_lease": lease,
    }
    storage = dict(ref["storage"])
    malformed_payloads = [
        {**payload, "dataset_manifest_ref": {**ref, "legacy": True}},
        {**payload, "dataset_manifest_ref": {**ref, "id": "dm_imp_foreign"}},
        {
            **payload,
            "dataset_manifest_ref": {
                **ref,
                "storage": {
                    **storage,
                    "manifest_path": (
                        "project_assets/ws_123/scope_123/imports/imp_foreign/manifest.json"
                    ),
                },
            },
        },
        {
            **payload,
            "object_key": "project_assets/ws_123/scope_123/imports/imp_123/other.json",
        },
        {
            **payload,
            "read_lease": {**lease, "dataset_manifest_ref_id": "dm_imp_foreign"},
        },
        {**payload, "dataset_manifest_ref": {**ref, "item_count": 4}},
        {
            **payload,
            "object_key": ("project_assets/ws_foreign/scope_123/imports/imp_123/manifest.json"),
            "dataset_manifest_ref": foreign_workspace_ref,
        },
    ]

    for malformed in malformed_payloads:
        with pytest.raises(RuntimeError):
            decode_dataset_session_response(
                malformed,
                project_id=PROJECT_ID,
                request_payload=_session_request_payload(),
            )


@pytest.mark.parametrize(
    "scope_segment",
    ["scope\tbad", "scope\x7fbad", "scope\u0085bad", "e\u0301", "scope_123 "],
)
def test_dataset_session_rejects_noncanonical_manifest_key_components(
    scope_segment: str,
) -> None:
    object_key = f"project_assets/ws_123/{scope_segment}/imports/imp_123/manifest.json"
    ref, lease = _ref_payload(object_key=object_key)
    payload = {
        "workspace_id": "ws_123",
        "project_id": PROJECT_ID,
        "import_id": IMPORT_ID,
        "status": "pending_upload",
        "object_key": object_key,
        "dataset_manifest_ref": ref,
        "read_lease": lease,
    }

    with pytest.raises(RuntimeError, match="canonical relative object path"):
        decode_dataset_session_response(
            payload,
            project_id=PROJECT_ID,
            request_payload=_session_request_payload(),
        )


@pytest.mark.parametrize(
    "import_id",
    [
        "imp_",
        "imp_a/b",
        "imp_a\\b",
        "imp_.",
        "imp_..",
        "imp_a%2Fb",
        "imp_a\x7fb",
        "imp_é",
        "imp_a b",
        f"imp_{'a' * 61}",
    ],
)
def test_dataset_session_rejects_import_id_outside_one_canonical_path_component(
    import_id: str,
) -> None:
    payload = _session_response(import_id=import_id)

    with pytest.raises(RuntimeError, match="one canonical imp_ path component"):
        decode_dataset_session_response(
            payload,
            project_id=PROJECT_ID,
            request_payload=_session_request_payload(),
        )


@pytest.mark.parametrize(
    "import_id",
    ["imp_-a", "imp__a", "imp_--", f"imp_{'a' * 60}"],
)
def test_dataset_session_accepts_every_declared_import_id_suffix_character(
    import_id: str,
) -> None:
    payload = _session_response(import_id=import_id)

    assert (
        decode_dataset_session_response(
            payload,
            project_id=PROJECT_ID,
            request_payload=_session_request_payload(),
        )
        is payload
    )


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
        "object_key": "project_assets/ws_123/scope_123/imports/imp_123/files/images/train/a.png",
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
        _decode_batch_urls(
            payload,
            requested_files=requested,
        )
        is payload
    )
    for files in ([item, item], [], [{**item, "relative_path": "extra.png"}]):
        with pytest.raises(RuntimeError):
            _decode_batch_urls(
                {**payload, "files": files},
                requested_files=requested,
            )
    with pytest.raises(RuntimeError, match="object_key does not match dataset session"):
        _decode_batch_urls(
            {**payload, "files": [{**item, "object_key": "foreign/import/files/a.png"}]},
            requested_files=requested,
        )
    with pytest.raises(RuntimeError, match="workspace_id does not match dataset session"):
        _decode_batch_urls(
            {**payload, "workspace_id": "ws_foreign"},
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
        "object_key": "project_assets/ws_123/scope_123/imports/imp_123/files/metadata/data.yaml",
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
        _decode_batch_urls(
            payload,
            requested_files=requested,
        )
        is payload
    )
    with pytest.raises(RuntimeError, match="Content-Type"):
        _decode_batch_urls(
            {
                **payload,
                "files": [
                    {
                        **item,
                        "required_headers": {"Content-Type": "text/yaml"},
                    }
                ],
            },
            requested_files=requested,
        )
    with pytest.raises(RuntimeError, match="request file fields must be exact"):
        _decode_batch_urls(
            payload,
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
        "object_key": "project_assets/ws_123/scope_123/imports/imp_123/files/images/train/a.png",
        "upload_url": "https://storage.example/a?signature=secret",
        "required_headers": {"Content-Type": content_type},
        "expires_in": 900,
    }

    with pytest.raises(RuntimeError, match="content_type"):
        _decode_batch_urls(
            {
                "workspace_id": "ws_123",
                "project_id": PROJECT_ID,
                "import_id": IMPORT_ID,
                "files": [item],
            },
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
            request_payload=_session_request_payload(),
            session_response=_session_response(),
            files=[
                {
                    "relative_path": "images/train/a.png",
                    "object_key": (
                        "project_assets/ws_123/scope_123/imports/imp_123/files/images/train/a.png"
                    ),
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
            request_payload=_session_request_payload(),
            session_response=_session_response(),
        )
        is payload
    )
    with pytest.raises(RuntimeError, match="uploaded_files"):
        decode_batch_complete_response(
            {**payload, "uploaded_files": 0},
            project_id=PROJECT_ID,
            import_id=IMPORT_ID,
            requested_paths=["images/train/a.png"],
            request_payload=_session_request_payload(),
            session_response=_session_response(),
        )
    for invalid_count in (True, 1.0, "1"):
        with pytest.raises(RuntimeError, match="uploaded_files"):
            decode_batch_complete_response(
                {**payload, "uploaded_files": invalid_count},
                project_id=PROJECT_ID,
                import_id=IMPORT_ID,
                requested_paths=["images/train/a.png"],
                request_payload=_session_request_payload(),
                session_response=_session_response(),
            )
    with pytest.raises(RuntimeError, match="fields must be exact"):
        decode_batch_complete_response(
            {**payload, "dataset_version_id": "dv_123"},
            project_id=PROJECT_ID,
            import_id=IMPORT_ID,
            requested_paths=["images/train/a.png"],
            request_payload=_session_request_payload(),
            session_response=_session_response(),
        )
    with pytest.raises(RuntimeError, match="workspace_id does not match dataset session"):
        decode_batch_complete_response(
            {**payload, "workspace_id": "ws_foreign"},
            project_id=PROJECT_ID,
            import_id=IMPORT_ID,
            requested_paths=["images/train/a.png"],
            request_payload=_session_request_payload(),
            session_response=_session_response(),
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
    }
    assert decode_complete_import_response(
        complete,
        project_id=PROJECT_ID,
        import_id=IMPORT_ID,
        request_payload=_session_request_payload(),
        session_response=_session_response(),
    )
    job = {
        "workspace_id": "ws_123",
        "project_id": PROJECT_ID,
        "import_id": IMPORT_ID,
        "status": "succeeded",
        "progress": {"phase": "succeeded"},
        "error": {},
        "dataset_validation": None,
        "dataset_version_id": "dsv_123",
        "version_ref": _materialized_version_ref(),
    }
    assert _decode_job(job)
    for alias in (
        "success",
        "completed",
        "done",
        "error",
        "cancelled",
        "SUCCEEDED",
        " succeeded ",
    ):
        with pytest.raises(RuntimeError, match="unsupported status"):
            _decode_job({**job, "status": alias})


def test_complete_decoder_binds_the_persisted_session_workspace_and_manifest_key() -> None:
    request_payload = _session_request_payload()
    session_response = _session_response(request_payload=request_payload)
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
    }
    changed_ref, _changed_lease = _ref_payload(
        object_key=("project_assets/ws_123/scope_123/imports/imp_123/replacement.json")
    )

    with pytest.raises(RuntimeError, match="exact project_assets workspace/scope import manifest"):
        decode_complete_import_response(
            {**complete, "dataset_manifest_ref": changed_ref},
            project_id=PROJECT_ID,
            import_id=IMPORT_ID,
            request_payload=request_payload,
            session_response=session_response,
        )

    with pytest.raises(RuntimeError, match="workspace_id"):
        decode_complete_import_response(
            {**complete, "workspace_id": "ws_foreign"},
            project_id=PROJECT_ID,
            import_id=IMPORT_ID,
            request_payload=request_payload,
            session_response=session_response,
        )


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
        decode_complete_import_response(
            payload,
            project_id=PROJECT_ID,
            import_id=IMPORT_ID,
            request_payload=_session_request_payload(),
            session_response=_session_response(),
        )


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
        "status": status,
        "progress": {"phase": status},
        "error": {},
        "dataset_validation": None,
    }
    if identity_mode == "null":
        job.update({"dataset_version_id": None, "version_ref": None})

    assert _decode_job(job) is job


@pytest.mark.parametrize("status", sorted(IMPORT_STATUSES - {"succeeded"}))
def test_prepublication_import_job_rejects_product_version_identity(status: str) -> None:
    job = {
        "workspace_id": "ws_123",
        "project_id": PROJECT_ID,
        "import_id": IMPORT_ID,
        "status": status,
        "progress": {"phase": status},
        "error": {},
        "dataset_validation": None,
        "dataset_version_id": "dv_123",
        "version_ref": {"dataset_version_id": "dv_123"},
    }

    with pytest.raises(RuntimeError, match="must not expose"):
        _decode_job(job)


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
        "dataset_version_id": "dsv_123",
        "version_ref": _materialized_version_ref(),
    }
    job[field] = value

    with pytest.raises(RuntimeError, match=field):
        _decode_job(job)


def test_succeeded_import_job_rejects_conflicting_version_reference_identity() -> None:
    job = {
        "workspace_id": "ws_123",
        "project_id": PROJECT_ID,
        "import_id": IMPORT_ID,
        "status": "succeeded",
        "progress": {"phase": "succeeded"},
        "error": {},
        "dataset_validation": None,
        "dataset_version_id": "dsv_123",
        "version_ref": _materialized_version_ref("dsv_other"),
    }

    with pytest.raises(RuntimeError, match="version_ref dataset_version_id"):
        _decode_job(job)


def test_succeeded_import_job_rejects_self_consistent_version_for_another_import() -> None:
    job = {
        "workspace_id": "ws_123",
        "project_id": PROJECT_ID,
        "import_id": IMPORT_ID,
        "status": "succeeded",
        "progress": {"phase": "succeeded"},
        "error": {},
        "dataset_validation": None,
        "dataset_version_id": "dsv_foreign",
        "version_ref": _materialized_version_ref("dsv_foreign"),
    }

    with pytest.raises(RuntimeError, match="import-derived identity"):
        _decode_job(job)


def test_succeeded_import_job_rejects_version_path_for_another_project_scope() -> None:
    job = {
        "workspace_id": "ws_123",
        "project_id": PROJECT_ID,
        "import_id": IMPORT_ID,
        "status": "succeeded",
        "progress": {"phase": "succeeded"},
        "error": {},
        "dataset_validation": None,
        "dataset_version_id": "dsv_123",
        "version_ref": _materialized_version_ref(project_scope_id="scope_foreign"),
    }

    with pytest.raises(RuntimeError, match="project_scope_id and dataset_version_id"):
        _decode_job(job)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda ref: ref.pop("storage_kind"),
        lambda ref: ref.update({"id": "dsv_123"}),
        lambda ref: ref.update({"storage_kind": "empty"}),
        lambda ref: ref.update({"lakefs_tag": "dsv_foreign"}),
        lambda ref: ref.update({"manifest_path": "foreign/manifest.json"}),
        lambda ref: ref.update({"content_digest": "a" * 64}),
        lambda ref: ref.update({"item_count": True}),
    ],
)
def test_succeeded_import_job_rejects_noncanonical_exact_version_ref(mutation) -> None:
    version_ref = _materialized_version_ref()
    mutation(version_ref)
    job = {
        "workspace_id": "ws_123",
        "project_id": PROJECT_ID,
        "import_id": IMPORT_ID,
        "status": "succeeded",
        "progress": {"phase": "succeeded"},
        "error": {},
        "dataset_validation": None,
        "dataset_version_id": "dsv_123",
        "version_ref": version_ref,
    }

    with pytest.raises(RuntimeError, match="version_ref"):
        _decode_job(job)


def test_import_job_rejects_workspace_drift_from_dataset_session() -> None:
    job = {
        "workspace_id": "ws_foreign",
        "project_id": PROJECT_ID,
        "import_id": IMPORT_ID,
        "status": "running",
        "progress": {"phase": "running"},
        "error": {},
        "dataset_validation": None,
    }

    with pytest.raises(RuntimeError, match="workspace_id does not match dataset session"):
        _decode_job(job)


def test_complete_decoder_accepts_the_shared_server_queued_contract() -> None:
    contract = _IMPORT_COMPLETE_CONTRACT
    payload = dict(contract["example"])
    ref = payload["dataset_manifest_ref"]
    assert isinstance(ref, dict)
    request_payload = _session_request_payload(
        format_name=str(ref["format"]),
        file_count=int(ref["item_count"]),
        total_bytes=int(ref["byte_count"]),
    )
    storage = ref["storage"]
    assert isinstance(storage, dict)
    session_response = _session_response(
        project_id=str(payload["project_id"]),
        import_id=str(payload["import_id"]),
        workspace_id=str(payload["workspace_id"]),
        request_payload=request_payload,
        object_key=str(storage["manifest_path"]),
    )

    assert set(payload) == set(contract["fields"])
    assert (
        decode_complete_import_response(
            payload,
            project_id=str(payload["project_id"]),
            import_id=str(payload["import_id"]),
            request_payload=request_payload,
            session_response=session_response,
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
                request_payload=request_payload,
                session_response=session_response,
            )
    for field in contract["object_fields"]:
        with pytest.raises(RuntimeError, match=field):
            decode_complete_import_response(
                {**payload, field: {}},
                project_id=str(payload["project_id"]),
                import_id=str(payload["import_id"]),
                request_payload=request_payload,
                session_response=session_response,
            )


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
