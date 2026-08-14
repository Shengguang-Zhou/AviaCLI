from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from avia_cli.commands.dataset import (
    _print_inspect_result,
    _print_verify_result,
    handle_dataset_command,
)
from avia_cli.commands.imports import handle_import_command
from avia_cli.core.uploads.inspect import (
    build_cleanup_plan,
    inspect_dataset,
    verify_dataset,
)
from avia_cli.parser import _build_parser


def _write_yolo_dataset(root: Path, *, with_label: bool = True) -> None:
    images = root / "images" / "train"
    labels = root / "labels" / "train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    (root / "classes.txt").write_text("aircraft\n", encoding="utf-8")
    Image.new("RGB", (16, 12)).save(images / "a.jpg")
    if with_label:
        (labels / "a.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")


def _materialized_version_ref(
    dataset_version_id: str = "dsv_done",
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
        "byte_count": 1,
    }


def test_dataset_parser_exposes_inspect_verify_and_cleanup_plan(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    _write_yolo_dataset(source)
    parser = _build_parser()

    inspect_args = parser.parse_args(
        [
            "dataset",
            "inspect",
            "--source",
            str(source),
            "--format",
            "yolo",
            "--task-key",
            "detect",
            "--json",
        ]
    )
    verify_args = parser.parse_args(
        [
            "dataset",
            "verify",
            "--source",
            str(source),
            "--format",
            "yolo",
            "--task-key",
            "detect",
            "--json",
        ]
    )
    cleanup_args = parser.parse_args(
        [
            "dataset",
            "cleanup-plan",
            "--api",
            "http://127.0.0.1:8080/api/v1",
            "--token",
            "avia_test",
            "--project",
            "proj_123456789abc",
            "--source",
            str(source),
            "--json",
        ]
    )

    assert inspect_args.dataset_command == "inspect"
    assert verify_args.dataset_command == "verify"
    assert cleanup_args.dataset_command == "cleanup-plan"


@pytest.mark.parametrize("command", ["inspect", "verify", "upload"])
@pytest.mark.parametrize("option", ["--max-files", "--max-samples"])
def test_dataset_commands_reject_historical_truncation_options(command: str, option: str) -> None:
    argv = ["dataset", command]
    if command == "upload":
        argv.extend(["--project", "proj_123456789abc"])
    argv.extend(
        [
            "--source",
            "/data/example",
            "--format",
            "yolo",
            "--task-key",
            "detect",
            option,
            "1",
        ]
    )

    with pytest.raises(SystemExit):
        _build_parser().parse_args(argv)


@pytest.mark.parametrize(
    "argv",
    [
        ["auth", "login", "--device-timeout", "0"],
        ["auth", "login", "--poll-interval", "0"],
        ["dataset", "cleanup-plan", "--project", "p", "--limit", "0"],
        ["dataset", "cleanup-plan", "--project", "p", "--limit", "201"],
        ["dataset", "upload", "--project", "p", "--concurrency", "0"],
        ["dataset", "upload", "--project", "p", "--batch-size", "1001"],
        ["dataset", "upload", "--project", "p", "--hash-workers", "0"],
        ["dataset", "upload", "--project", "p", "--stream-flush-size", "0"],
        ["dataset", "upload", "--project", "p", "--state-flush-every", "0"],
        ["dataset", "upload", "--project", "p", "--state-flush-interval", "0"],
        ["dataset", "upload", "--project", "p", "--upload-retries", "0"],
        ["dataset", "upload", "--project", "p", "--upload-connect-timeout", "0"],
        ["dataset", "upload", "--project", "p", "--poll-interval", "0"],
        ["auth", "login", "--poll-interval", "nan"],
        ["dataset", "upload", "--project", "p", "--progress-interval", "nan"],
        ["dataset", "upload", "--project", "p", "--upload-read-timeout", "inf"],
    ],
)
def test_numeric_cli_contracts_reject_invalid_values_before_execution(argv: list[str]) -> None:
    if argv[:2] == ["dataset", "upload"]:
        argv.extend(["--source", "/data/example", "--format", "yolo", "--task-key", "detect"])

    with pytest.raises(SystemExit):
        _build_parser().parse_args(argv)


@pytest.mark.parametrize(
    "argv",
    [
        [
            "import",
            "create",
            "--project",
            "proj_123456789abc",
            "--source",
            "s3://bucket/prefix",
        ],
        ["dataset", "scan", "--source", "/data/example"],
        ["dataset", "inspect", "--source", "/data/example"],
        ["dataset", "verify", "--source", "/data/example"],
        [
            "dataset",
            "upload",
            "--project",
            "proj_123456789abc",
            "--source",
            "/data/example",
        ],
    ],
)
def test_dataset_commands_require_explicit_format_and_task_key(argv: list[str]) -> None:
    parser = _build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(argv)


def test_dataset_inspect_returns_nonzero_for_a_real_corrupt_image(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "dataset"
    _write_yolo_dataset(source)
    (source / "images" / "train" / "a.jpg").write_bytes(b"not-an-image")
    args = _build_parser().parse_args(
        [
            "dataset",
            "inspect",
            "--source",
            str(source),
            "--format",
            "yolo",
            "--task-key",
            "detect",
            "--json",
        ]
    )

    assert handle_dataset_command(args) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed"
    assert payload["error_count"] == 1


def test_dataset_inspect_returns_zero_for_a_real_valid_image(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "dataset"
    _write_yolo_dataset(source)
    args = _build_parser().parse_args(
        [
            "dataset",
            "inspect",
            "--source",
            str(source),
            "--format",
            "yolo",
            "--task-key",
            "detect",
            "--json",
        ]
    )

    assert handle_dataset_command(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "ok"
    assert payload["error_count"] == 0


def test_dataset_upload_completes_local_validation_before_authentication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _build_parser().parse_args(
        [
            "dataset",
            "upload",
            "--project",
            "proj_123abc456def",
            "--source",
            str(tmp_path),
            "--format",
            "imagenet",
            "--task-key",
            "classify",
            "--class",
            "aircraft",
        ]
    )
    monkeypatch.setattr(
        "avia_cli.commands.dataset.token_from_args",
        lambda *_args, **_kwargs: pytest.fail("authentication ran before local validation"),
    )

    with pytest.raises(SystemExit, match="--class is only valid with --format yolo"):
        handle_dataset_command(args)


@pytest.mark.parametrize(
    "class_name",
    ["air\tcraft", "air\x7fcraft", "air\u0085craft", "e\u0301"],
)
def test_coco_class_catalog_fails_before_authentication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    class_name: str,
) -> None:
    images = tmp_path / "images"
    annotations = tmp_path / "annotations"
    images.mkdir()
    annotations.mkdir()
    Image.new("RGB", (16, 12)).save(images / "sample.png")
    (annotations / "instances.json").write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 1,
                        "file_name": "images/sample.png",
                        "width": 16,
                        "height": 12,
                    }
                ],
                "categories": [{"id": 1, "name": class_name}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [1, 2, 5, 4],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    args = _build_parser().parse_args(
        [
            "dataset",
            "upload",
            "--project",
            "proj_123abc456def",
            "--source",
            str(tmp_path),
            "--format",
            "coco",
            "--task-key",
            "detect",
        ]
    )
    monkeypatch.setattr(
        "avia_cli.commands.dataset.api_from_args",
        lambda *_args, **_kwargs: pytest.fail("API resolution ran before local validation"),
    )
    monkeypatch.setattr(
        "avia_cli.commands.dataset.token_from_args",
        lambda *_args, **_kwargs: pytest.fail("authentication ran before local validation"),
    )

    with pytest.raises(SystemExit, match="invalid_class_catalog"):
        handle_dataset_command(args)


@pytest.mark.parametrize(
    ("format_name", "task_key"),
    [
        ("yolo", "detect"),
        ("yolo", "classify"),
        ("yolo", "segment"),
        ("yolo", "pose"),
        ("yolo", "obb"),
        ("coco", "detect"),
        ("coco", "segment"),
        ("coco", "pose"),
        ("imagenet", "classify"),
        ("anomalib", "ad"),
    ],
)
def test_dataset_parser_accepts_exact_format_task_matrix(format_name: str, task_key: str) -> None:
    parser = _build_parser()

    args = parser.parse_args(
        [
            "dataset",
            "verify",
            "--source",
            "/data/example",
            "--format",
            format_name,
            "--task-key",
            task_key,
        ]
    )

    assert args.format == format_name
    assert args.task_key == task_key


@pytest.mark.parametrize(
    ("format_name", "task_key"),
    [
        ("yolo", "ad"),
        ("coco", "classify"),
        ("coco", "obb"),
        ("imagenet", "detect"),
        ("anomalib", "classify"),
    ],
)
def test_invalid_folder_upload_pair_fails_before_auth_or_http(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    format_name: str,
    task_key: str,
) -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "dataset",
            "upload",
            "--project",
            "proj_123456789abc",
            "--source",
            str(tmp_path),
            "--format",
            format_name,
            "--task-key",
            task_key,
        ]
    )
    monkeypatch.setattr(
        "avia_cli.commands.dataset.api_from_args",
        lambda _args: pytest.fail("invalid pair reached auth resolution"),
    )

    with pytest.raises(SystemExit, match=f"format '{format_name}'.*task '{task_key}'"):
        handle_dataset_command(args)


def test_invalid_source_import_pair_fails_before_auth_or_http(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parser = _build_parser()
    args = parser.parse_args(
        [
            "import",
            "create",
            "--project",
            "proj_123456789abc",
            "--source",
            "s3://bucket/prefix",
            "--format",
            "imagenet",
            "--task-key",
            "detect",
        ]
    )
    monkeypatch.setattr(
        "avia_cli.commands.imports.api_from_args",
        lambda _args: pytest.fail("invalid pair reached auth resolution"),
    )

    with pytest.raises(SystemExit, match=r"format 'imagenet'.*task 'detect'"):
        handle_import_command(args)


@pytest.mark.parametrize(
    "source",
    ["s3://bucket/prefix/", "/datasets/prefix/", "datasets/prefix", " datasets/prefix/"],
)
def test_source_import_rejects_noncanonical_object_prefix_before_auth(
    source: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _build_parser().parse_args(
        [
            "import",
            "create",
            "--project",
            "proj_123456789abc",
            "--source",
            source,
            "--format",
            "yolo",
            "--task-key",
            "detect",
        ]
    )
    monkeypatch.setattr(
        "avia_cli.commands.imports.api_from_args",
        lambda _args: pytest.fail("invalid object-prefix URI reached auth resolution"),
    )

    with pytest.raises(SystemExit, match="canonical bare object path"):
        handle_import_command(args)


@pytest.mark.parametrize(
    "classes",
    [
        ["plane", "plane"],
        [""],
        ["air\tcraft"],
        ["air\x7fcraft"],
        ["air\u0085craft"],
        ["e\u0301"],
    ],
)
def test_source_import_rejects_invalid_classes_before_auth(
    classes: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = [
        "import",
        "create",
        "--project",
        "proj_123456789abc",
        "--source",
        "datasets/prefix/",
        "--format",
        "yolo",
        "--task-key",
        "detect",
    ]
    for class_name in classes:
        argv.extend(["--class", class_name])
    args = _build_parser().parse_args(argv)
    monkeypatch.setattr(
        "avia_cli.commands.imports.api_from_args",
        lambda _args: pytest.fail("invalid source-import classes reached auth resolution"),
    )

    with pytest.raises(RuntimeError, match="source-import request classes"):
        handle_import_command(args)


def test_anomalib_source_import_submits_canonical_binary_taxonomy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _build_parser().parse_args(
        [
            "import",
            "create",
            "--project",
            "proj_123456789abc",
            "--source",
            "datasets/mvtec-bottle/",
            "--format",
            "anomalib",
            "--task-key",
            "ad",
        ]
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        "avia_cli.commands.imports.api_from_args",
        lambda _args: "https://avia.example/api/v1",
    )
    monkeypatch.setattr(
        "avia_cli.commands.imports.token_from_args",
        lambda _args, *, api: "token",
    )

    def create_source_import(**kwargs):
        captured.update(kwargs)
        return {"status": "queued"}

    monkeypatch.setattr(
        "avia_cli.commands.imports.create_source_import",
        create_source_import,
    )

    assert handle_import_command(args) == 0
    assert captured["payload"] == {
        "source_kind": "object_prefix",
        "uri": "datasets/mvtec-bottle/",
        "format": "anomalib",
        "task_key": "ad",
        "classes": ["good", "bad"],
        "auto_post_processing": True,
    }


def test_anomalib_source_import_rejects_user_class_override_before_auth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _build_parser().parse_args(
        [
            "import",
            "create",
            "--project",
            "proj_123456789abc",
            "--source",
            "datasets/mvtec-bottle/",
            "--format",
            "anomalib",
            "--task-key",
            "ad",
            "--class",
            "good",
        ]
    )
    monkeypatch.setattr(
        "avia_cli.commands.imports.api_from_args",
        lambda _args: pytest.fail("invalid class override reached auth resolution"),
    )

    with pytest.raises(SystemExit, match="only valid with --format yolo"):
        handle_import_command(args)


def test_human_readable_inspect_and_verify_output_carries_task_key(
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary = {
        "format": "yolo",
        "task_key": "pose",
        "file_count": 3,
        "image_count": 1,
        "label_count": 1,
        "mask_count": 0,
        "total_bytes": 100,
    }

    _print_inspect_result(summary, json_output=False)
    _print_verify_result(
        {
            **summary,
            "status": "ok",
            "error_count": 0,
            "warning_count": 0,
        },
        json_output=False,
    )

    output = capsys.readouterr().out
    assert "yolo/pose dataset" in output
    assert "dataset verify yolo/pose ok" in output


def test_inspect_dataset_returns_compact_manifest_summary(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    _write_yolo_dataset(source)

    result = inspect_dataset(source=source, format_name="yolo", task_key="detect")

    assert result["format"] == "yolo"
    assert result["task_key"] == "detect"
    assert result["file_count"] == 3
    assert result["image_count"] == 1
    assert result["label_count"] == 1
    assert result["classes"] == ["aircraft"]
    assert "files" not in result


def test_verify_dataset_fails_on_missing_yolo_labels(
    tmp_path: Path,
) -> None:
    source = tmp_path / "dataset"
    _write_yolo_dataset(source, with_label=False)

    result = verify_dataset(source=source, format_name="yolo", task_key="detect")

    assert result["status"] == "failed"
    assert result["error_count"] == 1
    assert result["errors"][0]["code"] == "missing_yolo_label"
    assert result["errors"][0]["path"] == "images/train/a.jpg"
    assert result["warnings"] == []


def test_verify_dataset_fails_when_yolo_has_no_images(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    (source / "labels").mkdir(parents=True)
    (source / "labels" / "orphan.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    result = verify_dataset(source=source, format_name="yolo", task_key="detect")

    assert result["status"] == "failed"
    assert any(error["code"] == "no_images" for error in result["errors"])


def test_cleanup_plan_uses_yolotaskcv_api_and_local_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "dataset"
    _write_yolo_dataset(source)
    state_dir = source / ".avia" / "imports"
    project_state = state_dir / "proj_123456789abc"
    project_state.mkdir(parents=True)
    idempotency_key = "5d74e1c1-f1e4-4b4b-9b42-cae872f71c4a"
    state_path = project_state / f"{idempotency_key}.json"
    session_payload = {
        "idempotency_key": idempotency_key,
        "format": "yolo",
        "root_name": "dataset",
        "task_key": "detect",
        "classes": ["aircraft"],
        "file_count": 1,
        "total_bytes": 1,
    }
    manifest_ref = {
        "id": "dm_imp_done",
        "format": "yolo",
        "item_count": 1,
        "byte_count": 1,
        "storage": {
            "kind": "minio",
            "manifest_path": ("project_assets/ws_123/scope_123/imports/imp_done/manifest.json"),
            "path_prefix": "project_assets/ws_123/scope_123/imports/imp_done",
            "lakefs_repo": None,
            "lakefs_commit": None,
            "dataset_version_id": None,
        },
    }
    read_lease = {
        "id": "lease_imp_done",
        "scope": "read",
        "access": "object_ref",
        "dataset_manifest_ref_id": "dm_imp_done",
    }
    state_path.write_text(
        json.dumps(
            {
                "schema_version": 6,
                "api": "http://127.0.0.1:8080/api/v1",
                "phase": "completed",
                "project_id": "proj_123456789abc",
                "import_id": "imp_done",
                "idempotency_key": idempotency_key,
                "source": str(source.resolve()),
                "format": "yolo",
                "task_key": "detect",
                "session_payload": session_payload,
                "session_response": {
                    "workspace_id": "ws_123",
                    "project_id": "proj_123456789abc",
                    "import_id": "imp_done",
                    "status": "pending_upload",
                    "object_key": (
                        "project_assets/ws_123/scope_123/imports/imp_done/manifest.json"
                    ),
                    "dataset_manifest_ref": manifest_ref,
                    "read_lease": read_lease,
                },
                "complete_response": {
                    "workspace_id": "ws_123",
                    "project_id": "proj_123456789abc",
                    "import_id": "imp_done",
                    "status": "queued",
                    "dataset_manifest_ref": manifest_ref,
                    "read_lease": read_lease,
                    "reason": "queued",
                    "dispatch_mode": "celery",
                    "worker_task_id": "task_123",
                },
                "files": {
                    "classes.txt": {
                        "uploaded": True,
                        "streamed": True,
                        "size_bytes": 1,
                        "sha256": "a" * 64,
                        "width": 0,
                        "height": 0,
                        "content_type": "text/plain",
                        "object_key": (
                            "project_assets/ws_123/scope_123/imports/imp_done/files/classes.txt"
                        ),
                        "version_id": "version-classes",
                        "source_identity": {
                            "device": 1,
                            "inode": 1,
                            "size_bytes": 1,
                            "mtime_ns": 1,
                            "ctime_ns": 1,
                        },
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    calls: list[dict[str, object]] = []

    def fake_request_json_with_retries(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        return {
            "imports": [
                {
                    "import_id": "imp_done",
                    "status": "succeeded",
                    "job_type": "dataset.import.yolo",
                    "object_key": (
                        "project_assets/ws_123/scope_123/imports/imp_done/manifest.json"
                    ),
                    "progress": {"phase": "done"},
                    "error": {},
                    "dataset_validation": None,
                    "dataset_version_id": "dsv_done",
                    "version_ref": _materialized_version_ref(),
                    "created_at": "2026-07-15T10:00:00+00:00",
                    "updated_at": "2026-07-15T10:01:00+00:00",
                }
            ],
            "next_cursor": None,
            "project_id": "proj_123456789abc",
        }

    monkeypatch.setattr(
        "avia_cli.core.uploads.inspect._request_json_with_retries",
        fake_request_json_with_retries,
    )

    result = build_cleanup_plan(
        api="http://127.0.0.1:8080/api/v1",
        token="avia_test",
        project_id="proj_123456789abc",
        source=source,
        state_dir=None,
        limit=20,
    )

    assert calls
    assert calls[0]["method"] == "GET"
    assert (
        calls[0]["url"]
        == "http://127.0.0.1:8080/api/v1/projects/proj_123456789abc/ingestion-jobs?limit=20"
    )
    assert result["storage_boundary"] == "server_owned"
    assert result["server_imports"][0]["import_id"] == "imp_done"
    assert result["local_states"][0]["state_path"] == str(state_path)
    assert result["local_states"][0]["task_key"] == "detect"
    assert result["actions"] == [
        {
            "kind": "remove_local_state",
            "path": str(state_path),
            "reason": "server import is terminal and local resume state is completed",
        }
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda item: item.pop("job_type"),
        lambda item: item.update({"legacy_source": "archive"}),
        lambda item: item.update({"progress": None}),
        lambda item: item.update({"object_key": "imports/imp_done/manifest.json"}),
        lambda item: item.update({"version_ref": {"id": "dv_123"}}),
        lambda item: item.update({"version_ref": {"dataset_version_id": ""}}),
        lambda item: item.update({"version_ref": {"dataset_version_id": "dv_other"}}),
        lambda item: item.update(
            {
                "dataset_version_id": "dsv_other",
                "version_ref": _materialized_version_ref("dsv_other"),
            }
        ),
        lambda item: item.update(
            {"version_ref": _materialized_version_ref(project_scope_id="scope_foreign")}
        ),
    ],
)
def test_cleanup_plan_rejects_noncanonical_ingestion_job_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation
) -> None:
    entry = {
        "import_id": "imp_done",
        "status": "succeeded",
        "job_type": "dataset.import.yolo",
        "object_key": "project_assets/ws_123/scope_123/imports/imp_done/manifest.json",
        "progress": {"phase": "done"},
        "error": {},
        "dataset_validation": None,
        "dataset_version_id": "dsv_done",
        "version_ref": _materialized_version_ref(),
        "created_at": "2026-07-15T10:00:00+00:00",
        "updated_at": "2026-07-15T10:01:00+00:00",
    }
    mutation(entry)
    monkeypatch.setattr(
        "avia_cli.core.uploads.inspect._request_json_with_retries",
        lambda **_kwargs: {
            "imports": [entry],
            "next_cursor": None,
            "project_id": "proj_123456789abc",
        },
    )

    with pytest.raises(RuntimeError, match="ingestion-jobs entry"):
        build_cleanup_plan(
            api="http://127.0.0.1:8080/api/v1",
            token="token",
            project_id="proj_123456789abc",
            state_dir=tmp_path / "state",
        )


def test_cleanup_plan_rejects_corrupt_local_state_with_exact_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state_dir = tmp_path / "state"
    path = state_dir / "proj_123456789abc" / "broken.json"
    path.parent.mkdir(parents=True)
    path.write_text("{", encoding="utf-8")
    monkeypatch.setattr(
        "avia_cli.core.uploads.inspect._request_json_with_retries",
        lambda **_kwargs: {
            "imports": [],
            "next_cursor": None,
            "project_id": "proj_123456789abc",
        },
    )

    with pytest.raises(SystemExit, match=rf"invalid cleanup state {path}"):
        build_cleanup_plan(
            api="http://127.0.0.1:8080/api/v1",
            token="token",
            project_id="proj_123456789abc",
            state_dir=state_dir,
        )
