from __future__ import annotations

import json
from pathlib import Path

import pytest

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
    (images / "a.jpg").write_bytes(b"not-a-real-image")
    if with_label:
        (labels / "a.txt").write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")


def test_dataset_parser_exposes_inspect_verify_and_cleanup_plan(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    _write_yolo_dataset(source)
    parser = _build_parser()

    inspect_args = parser.parse_args(
        ["dataset", "inspect", "--source", str(source), "--format", "yolo", "--json"]
    )
    verify_args = parser.parse_args(
        ["dataset", "verify", "--source", str(source), "--format", "yolo", "--json"]
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


def test_inspect_dataset_returns_compact_manifest_summary(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    _write_yolo_dataset(source)

    result = inspect_dataset(source=source, format_name="yolo")

    assert result["format"] == "yolo"
    assert result["file_count"] == 3
    assert result["image_count"] == 1
    assert result["label_count"] == 1
    assert result["classes"] == ["aircraft"]
    assert "files" not in result


def test_verify_dataset_reports_missing_yolo_labels_without_failing_upload(
    tmp_path: Path,
) -> None:
    source = tmp_path / "dataset"
    _write_yolo_dataset(source, with_label=False)

    result = verify_dataset(source=source, format_name="yolo")

    assert result["status"] == "ok"
    assert result["error_count"] == 0
    warnings = result["warnings"]
    assert warnings
    assert warnings[0]["code"] == "missing_yolo_label"
    assert warnings[0]["path"] == "images/train/a.jpg"


def test_verify_dataset_fails_when_yolo_has_no_images(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    (source / "labels").mkdir(parents=True)
    (source / "labels" / "orphan.txt").write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    result = verify_dataset(source=source, format_name="yolo")

    assert result["status"] == "failed"
    assert result["error_count"] == 1
    assert result["errors"][0]["code"] == "no_images"


def test_cleanup_plan_uses_yolotaskcv_api_and_local_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "dataset"
    _write_yolo_dataset(source)
    state_dir = source / ".avia" / "imports"
    project_state = state_dir / "proj_123456789abc"
    project_state.mkdir(parents=True)
    state_path = project_state / "imp_done.json"
    state_path.write_text(
        json.dumps(
            {
                "project_id": "proj_123456789abc",
                "import_id": "imp_done",
                "source": str(source.resolve()),
                "format": "yolo",
                "completed": True,
                "files": {
                    "images/train/a.jpg": {"uploaded": True, "streamed": True},
                    "labels/train/a.txt": {"uploaded": True, "streamed": True},
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
                    "progress": {"phase": "done"},
                }
            ],
            "next_cursor": None,
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
    assert result["actions"] == [
        {
            "kind": "remove_local_state",
            "path": str(state_path),
            "reason": "server import is terminal and local resume state is completed",
        }
    ]
