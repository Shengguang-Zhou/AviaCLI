from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from avia_cli.core.uploads.dataset import prepare_dataset_upload, upload_prepared_dataset
from avia_cli.core.uploads.timing import UploadTimingRecorder
from avia_cli.parser import _build_parser

PROJECT_ID = "proj_123456789abc"
IMPORT_ID = "imp_timing_failure"


def _remote_identity(
    request_payload: dict[str, object],
) -> tuple[str, dict[str, object], dict[str, object]]:
    object_key = f"project_assets/ws_123/scope_123/imports/{IMPORT_ID}/manifest.json"
    ref_id = f"dm_{IMPORT_ID}"
    ref = {
        "id": ref_id,
        "format": request_payload["format"],
        "item_count": request_payload["file_count"],
        "byte_count": request_payload["total_bytes"],
        "storage": {
            "kind": "minio",
            "manifest_path": object_key,
            "path_prefix": f"project_assets/ws_123/scope_123/imports/{IMPORT_ID}",
            "lakefs_repo": None,
            "lakefs_commit": None,
            "dataset_version_id": None,
        },
    }
    lease = {
        "id": f"lease_{IMPORT_ID}",
        "scope": "read",
        "access": "object_ref",
        "dataset_manifest_ref_id": ref_id,
    }
    return object_key, ref, lease


def _session_response(request_payload: dict[str, object]) -> dict[str, object]:
    object_key, ref, lease = _remote_identity(request_payload)
    return {
        "workspace_id": "ws_123",
        "project_id": PROJECT_ID,
        "import_id": IMPORT_ID,
        "status": "pending_upload",
        "object_key": object_key,
        "dataset_manifest_ref": ref,
        "read_lease": lease,
    }


def _complete_response(request_payload: dict[str, object]) -> dict[str, object]:
    _object_key, ref, lease = _remote_identity(request_payload)
    return {
        "workspace_id": "ws_123",
        "project_id": PROJECT_ID,
        "import_id": IMPORT_ID,
        "status": "queued",
        "dataset_manifest_ref": ref,
        "read_lease": lease,
        "reason": "queued",
        "dispatch_mode": "celery",
        "worker_task_id": f"avia-import:{IMPORT_ID}:1",
    }


def _prepare_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[object, Path, list[str]]:
    source = tmp_path / "dataset"
    image = source / "images" / "train" / "sample.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (16, 12)).save(image)
    label = source / "labels" / "train" / "sample.txt"
    label.parent.mkdir(parents=True)
    label.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    (source / "classes.txt").write_text("aircraft\n", encoding="utf-8")
    state_dir = tmp_path / "state"
    args = _build_parser().parse_args(
        [
            "dataset",
            "upload",
            "--project",
            PROJECT_ID,
            "--source",
            str(source),
            "--format",
            "yolo",
            "--task-key",
            "detect",
            "--concurrency",
            "3",
            "--batch-size",
            "100",
            "--hash-workers",
            "1",
            "--batch-complete-concurrency",
            "1",
            "--stream-flush-size",
            "100",
            "--state-dir",
            str(state_dir),
        ]
    )
    remote_effects: list[str] = []
    put_barrier = threading.Barrier(3)
    monkeypatch.setattr(
        "avia_cli.core.uploads.dataset._create_dataset_session",
        lambda **kwargs: _session_response(kwargs["payload"]),
    )

    def sign(**kwargs: object) -> dict[str, object]:
        files = kwargs["files"]
        assert isinstance(files, list)
        return {
            "files": [
                {
                    **item,
                    "content_type": "application/octet-stream",
                    "upload_url": f"https://objects.example/{item['relative_path']}",
                    "object_key": (
                        "project_assets/ws_123/scope_123/imports/"
                        f"{IMPORT_ID}/files/{item['relative_path']}"
                    ),
                    "required_headers": {"Content-Type": "application/octet-stream"},
                }
                for item in files
            ]
        }

    monkeypatch.setattr("avia_cli.core.uploads.dataset._batch_upload_urls", sign)
    monkeypatch.setattr(
        "avia_cli.core.uploads.dataset._resolve_transport_concurrency",
        lambda *_args, **_kwargs: None,
    )

    def put(**kwargs: object) -> None:
        remote_effects.append(f"put:{Path(str(kwargs['path'])).name}")
        put_barrier.wait(timeout=2)

    monkeypatch.setattr("avia_cli.core.uploads.dataset._put_file_with_retries", put)
    monkeypatch.setattr(
        "avia_cli.core.uploads.dataset._complete_dataset_file_batch",
        lambda **_kwargs: remote_effects.append("batch-complete") or {"status": "ok"},
    )
    monkeypatch.setattr(
        "avia_cli.core.uploads.dataset._complete_import",
        lambda **kwargs: (
            remote_effects.append("import-finalize")
            or _complete_response(kwargs["request_payload"])
        ),
    )
    return args, state_dir, remote_effects


def _fail_timing_name(
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    original = UploadTimingRecorder.record

    def record(
        self: UploadTimingRecorder,
        name: str,
        *,
        duration_sec: float,
        file_count: int = 0,
        byte_count: int = 0,
    ) -> None:
        if name == target:
            raise OSError(f"{target} timing sink failed")
        original(
            self,
            name,
            duration_sec=duration_sec,
            file_count=file_count,
            byte_count=byte_count,
        )

    monkeypatch.setattr(UploadTimingRecorder, "record", record)


def _run(args: object) -> dict[str, Any]:
    prepared = prepare_dataset_upload(args)
    return upload_prepared_dataset(
        args,
        api="https://avia.example/api/v1",
        token="token",
        prepared=prepared,
    )


def _state(state_dir: Path) -> dict[str, Any]:
    path = next(state_dir.rglob("*.json"))
    return json.loads(path.read_text(encoding="utf-8"))


def test_put_timing_failure_persists_every_completed_put(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, state_dir, remote_effects = _prepare_upload(tmp_path, monkeypatch)
    _fail_timing_name(monkeypatch, "file_put")

    with pytest.raises(RuntimeError, match="file-put-telemetry"):
        _run(args)

    state = _state(state_dir)
    assert len([effect for effect in remote_effects if effect.startswith("put:")]) == 3
    assert all(item["uploaded"] is True for item in state["files"].values())


def test_batch_timing_failure_persists_every_completed_batch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, state_dir, remote_effects = _prepare_upload(tmp_path, monkeypatch)
    _fail_timing_name(monkeypatch, "batch_complete")

    with pytest.raises(RuntimeError, match="batch-complete-telemetry"):
        _run(args)

    state = _state(state_dir)
    assert remote_effects.count("batch-complete") == 1
    assert all(item["streamed"] is True for item in state["files"].values())


def test_finalize_timing_failure_persists_terminal_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args, state_dir, remote_effects = _prepare_upload(tmp_path, monkeypatch)
    _fail_timing_name(monkeypatch, "import_finalize")

    with pytest.raises(RuntimeError, match="import-finalize-telemetry"):
        _run(args)

    state = _state(state_dir)
    assert remote_effects.count("import-finalize") == 1
    assert state["phase"] == "completed"
    assert state["complete_response"]["status"] == "queued"
