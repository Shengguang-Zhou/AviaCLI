from __future__ import annotations

import json
import hashlib
import os
import re
import threading
import time
from pathlib import Path

import pytest
from PIL import Image

from avia_cli.core.uploads.dataset import (
    _assert_resume_source_unchanged,
    _capture_manifest_identities,
    upload_dataset,
)
from avia_cli.core.uploads.source_file import SourceFileChangedError
from avia_cli.core.uploads.state import _exclusive_upload_state_lock, _load_resume_state
from avia_cli.parser import _build_parser


def _write_state(
    state_dir: Path,
    *,
    project_id: str = "proj_123456789abc",
    import_id: str = "imp_123",
    task_key: str = "detect",
    idempotency_key: str = "5d74e1c1-f1e4-4b4b-9b42-cae872f71c4a",
) -> Path:
    path = state_dir / project_id / f"{idempotency_key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 3,
                "api": "https://avia.eurekailab.com/api/v1",
                "phase": "uploading",
                "project_id": project_id,
                "import_id": import_id,
                "idempotency_key": idempotency_key,
                "source": "/data/coco8",
                "format": "yolo",
                "task_key": task_key,
                "session_payload": {
                    "idempotency_key": idempotency_key,
                    "format": "yolo",
                    "root_name": "coco8",
                    "task_key": task_key,
                    "classes": ["aircraft"],
                    "file_count": 1,
                    "total_bytes": 1,
                },
                "complete_response": None,
                "files": {
                    "classes.txt": {
                        "uploaded": False,
                        "streamed": False,
                        "size_bytes": 1,
                        "sha256": "",
                        "width": 0,
                        "height": 0,
                        "object_key": None,
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
    return path


def test_resume_state_identity_includes_task_key(tmp_path: Path) -> None:
    _write_state(tmp_path, task_key="pose")

    state = _load_resume_state(
        state_dir=tmp_path,
        project_id="proj_123456789abc",
        api="https://avia.eurekailab.com/api/v1",
        source="/data/coco8",
        import_format="yolo",
        task_key="pose",
    )

    assert state is not None
    assert state["task_key"] == "pose"


def test_resume_rejects_completed_state_with_historical_version_reference(
    tmp_path: Path,
) -> None:
    path = _write_state(tmp_path)
    state = json.loads(path.read_text(encoding="utf-8"))
    state["phase"] = "completed"
    state["complete_response"] = {
        "workspace_id": "ws_123",
        "project_id": "proj_123456789abc",
        "import_id": "imp_123",
        "status": "queued",
        "dataset_manifest_ref": {"id": "dm_123"},
        "read_lease": {"id": "lease_123"},
        "reason": "queued",
        "dispatch_mode": "celery",
        "worker_task_id": "task_123",
        "dataset_version_id": "dv_123",
        "version_ref": {"id": "dv_123"},
    }
    path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(SystemExit, match="version_ref dataset_version_id"):
        _load_resume_state(
            state_dir=tmp_path,
            project_id="proj_123456789abc",
            api="https://avia.eurekailab.com/api/v1",
            source="/data/coco8",
            import_format="yolo",
            task_key="detect",
        )


def test_resume_scans_past_newer_near_match_to_find_unique_exact_task(
    tmp_path: Path,
) -> None:
    detect_path = _write_state(tmp_path, import_id="imp_detect", task_key="detect")
    pose_path = _write_state(
        tmp_path,
        import_id="imp_pose",
        task_key="pose",
        idempotency_key="6d74e1c1-f1e4-4b4b-9b42-cae872f71c4b",
    )
    newer = detect_path.stat().st_mtime_ns + 1_000_000_000
    os.utime(pose_path, ns=(newer, newer))

    state = _load_resume_state(
        state_dir=tmp_path,
        project_id="proj_123456789abc",
        api="https://avia.eurekailab.com/api/v1",
        source="/data/coco8",
        import_format="yolo",
        task_key="detect",
    )

    assert state is not None
    assert state["import_id"] == "imp_detect"


def test_resume_rejects_multiple_exact_states_as_ambiguous(tmp_path: Path) -> None:
    _write_state(tmp_path, import_id="imp_first", task_key="detect")
    _write_state(
        tmp_path,
        import_id="imp_second",
        task_key="detect",
        idempotency_key="6d74e1c1-f1e4-4b4b-9b42-cae872f71c4b",
    )

    with pytest.raises(SystemExit, match=r"ambiguous resume state"):
        _load_resume_state(
            state_dir=tmp_path,
            project_id="proj_123456789abc",
            api="https://avia.eurekailab.com/api/v1",
            source="/data/coco8",
            import_format="yolo",
            task_key="detect",
        )


def test_resume_ignores_completed_history_when_one_active_exact_state_exists(
    tmp_path: Path,
) -> None:
    completed_path = _write_state(tmp_path, import_id="imp_completed", task_key="detect")
    completed = json.loads(completed_path.read_text(encoding="utf-8"))
    completed["phase"] = "completed"
    completed["complete_response"] = {
        "workspace_id": "ws_123",
        "project_id": "proj_123456789abc",
        "import_id": "imp_completed",
        "status": "queued",
        "dataset_manifest_ref": {"id": "dm_completed"},
        "read_lease": {
            "id": "lease_completed",
            "dataset_manifest_ref_id": "dm_completed",
        },
        "reason": "queued",
        "dispatch_mode": "celery",
        "worker_task_id": "task_completed",
        "dataset_version_id": "dsv_completed",
        "version_ref": {"dataset_version_id": "dsv_completed"},
    }
    completed_path.write_text(json.dumps(completed), encoding="utf-8")
    _write_state(
        tmp_path,
        import_id="imp_active",
        task_key="detect",
        idempotency_key="6d74e1c1-f1e4-4b4b-9b42-cae872f71c4b",
    )

    state = _load_resume_state(
        state_dir=tmp_path,
        project_id="proj_123456789abc",
        api="https://avia.eurekailab.com/api/v1",
        source="/data/coco8",
        import_format="yolo",
        task_key="detect",
    )

    assert state is not None
    assert state["import_id"] == "imp_active"


def test_upload_state_lock_rejects_a_second_writer_for_the_same_dataset(
    tmp_path: Path,
) -> None:
    kwargs = {
        "state_dir": tmp_path,
        "project_id": "proj_123456789abc",
        "source": "/data/coco8",
        "import_format": "yolo",
        "task_key": "detect",
    }

    with _exclusive_upload_state_lock(**kwargs) as lock_path:
        assert lock_path.is_file()
        lock_record = json.loads(lock_path.read_text(encoding="utf-8"))
        assert lock_record == {
            "format": "yolo",
            "pid": lock_record["pid"],
            "project_id": "proj_123456789abc",
            "source": "/data/coco8",
            "task_key": "detect",
        }
        assert isinstance(lock_record["pid"], int)
        assert lock_record["pid"] > 0

        with pytest.raises(SystemExit, match=r'upload already active.*"pid":.*coco8'):
            with _exclusive_upload_state_lock(**kwargs):
                pytest.fail("a second writer acquired the same upload state lock")


def test_upload_state_lock_identity_separates_tasks(tmp_path: Path) -> None:
    shared = {
        "state_dir": tmp_path,
        "project_id": "proj_123456789abc",
        "source": "/data/coco8",
        "import_format": "yolo",
    }

    with _exclusive_upload_state_lock(**shared, task_key="detect"):
        with _exclusive_upload_state_lock(**shared, task_key="pose"):
            pass


def test_resume_rejects_state_from_another_task(tmp_path: Path) -> None:
    _write_state(tmp_path, task_key="detect")

    with pytest.raises(SystemExit, match=r"resume state task mismatch.*detect.*pose"):
        _load_resume_state(
            state_dir=tmp_path,
            project_id="proj_123456789abc",
            api="https://avia.eurekailab.com/api/v1",
            source="/data/coco8",
            import_format="yolo",
            task_key="pose",
        )


def test_resume_rejects_malformed_state_instead_of_skipping_it(tmp_path: Path) -> None:
    path = tmp_path / "proj_123456789abc" / "imp_bad.json"
    path.parent.mkdir(parents=True)
    path.write_text("{", encoding="utf-8")

    with pytest.raises(SystemExit, match="invalid resume state"):
        _load_resume_state(
            state_dir=tmp_path,
            project_id="proj_123456789abc",
            api="https://avia.eurekailab.com/api/v1",
            source="/data/coco8",
            import_format="yolo",
            task_key="detect",
        )


def test_resume_rejects_state_filename_that_disagrees_with_idempotency_key(
    tmp_path: Path,
) -> None:
    path = _write_state(tmp_path)
    wrong_path = path.with_name("wrong.json")
    path.rename(wrong_path)

    with pytest.raises(SystemExit, match=r"state filename must match idempotency_key"):
        _load_resume_state(
            state_dir=tmp_path,
            project_id="proj_123456789abc",
            api="https://avia.eurekailab.com/api/v1",
            source="/data/coco8",
            import_format="yolo",
            task_key="detect",
        )


def test_resume_rejects_historical_state_without_exact_schema(tmp_path: Path) -> None:
    path = tmp_path / "proj_123456789abc" / "imp_legacy.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "project_id": "proj_123456789abc",
                "import_id": "imp_legacy",
                "source": "/data/coco8",
                "format": "yolo",
                "task_key": "detect",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match=r"invalid resume state.*state fields must be exact"):
        _load_resume_state(
            state_dir=tmp_path,
            project_id="proj_123456789abc",
            api="https://avia.eurekailab.com/api/v1",
            source="/data/coco8",
            import_format="yolo",
            task_key="detect",
        )


def test_resume_rejects_state_with_noncanonical_session_payload(tmp_path: Path) -> None:
    path = _write_state(tmp_path)
    state = json.loads(path.read_text(encoding="utf-8"))
    state["session_payload"]["auto_crop_embedding"] = True
    path.write_text(json.dumps(state), encoding="utf-8")

    with pytest.raises(SystemExit, match=r"invalid resume state.*session_payload fields"):
        _load_resume_state(
            state_dir=tmp_path,
            project_id="proj_123456789abc",
            api="https://avia.eurekailab.com/api/v1",
            source="/data/coco8",
            import_format="yolo",
            task_key="detect",
        )


def test_resume_state_is_bound_to_the_exact_api_origin(tmp_path: Path) -> None:
    _write_state(tmp_path)

    with pytest.raises(SystemExit, match=r"resume state API mismatch.*avia\.eurekailab\.com"):
        _load_resume_state(
            state_dir=tmp_path,
            project_id="proj_123456789abc",
            api="https://another.example.test/api/v1",
            source="/data/coco8",
            import_format="yolo",
            task_key="detect",
        )


def test_manifest_identity_capture_accepts_empty_negative_label(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    label = source / "labels" / "train" / "negative.txt"
    label.parent.mkdir(parents=True)
    label.write_bytes(b"")

    identities = _capture_manifest_identities(
        source,
        [{"relative_path": "labels/train/negative.txt", "size_bytes": 0}],
    )

    assert identities["labels/train/negative.txt"]["size_bytes"] == 0


def test_resume_rejects_changed_source_identity_before_any_network_call(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    label = source / "labels" / "train" / "sample.txt"
    label.parent.mkdir(parents=True)
    label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    files = [{"relative_path": "labels/train/sample.txt", "size_bytes": label.stat().st_size}]
    original = _capture_manifest_identities(source, files)
    state = {
        "files": {
            "labels/train/sample.txt": {
                "uploaded": False,
                "streamed": False,
                "size_bytes": label.stat().st_size,
                "sha256": "",
                "width": 0,
                "height": 0,
                "source_identity": dict(original["labels/train/sample.txt"]),
            }
        }
    }
    label.write_text("0 0.4 0.5 0.2 0.2\n", encoding="utf-8")
    current = _capture_manifest_identities(source, files)

    with pytest.raises(SourceFileChangedError, match="resume source identity changed"):
        _assert_resume_source_unchanged(
            source_root=source,
            files=files,
            source_identities=current,
            state=state,
        )


def test_resume_rehashes_every_uploaded_file_and_rejects_digest_mismatch(
    tmp_path: Path,
) -> None:
    source = tmp_path / "dataset"
    label = source / "labels" / "train" / "sample.txt"
    label.parent.mkdir(parents=True)
    payload = b"0 0.5 0.5 0.2 0.2\n"
    label.write_bytes(payload)
    files = [{"relative_path": "labels/train/sample.txt", "size_bytes": len(payload)}]
    identities = _capture_manifest_identities(source, files)
    state = {
        "files": {
            "labels/train/sample.txt": {
                "uploaded": True,
                "streamed": False,
                "size_bytes": len(payload),
                "sha256": hashlib.sha256(b"different bytes").hexdigest(),
                "width": 0,
                "height": 0,
                "source_identity": dict(identities["labels/train/sample.txt"]),
                "object_key": "datasets/sample.txt",
            }
        }
    }

    with pytest.raises(SourceFileChangedError, match="uploaded file digest mismatch"):
        _assert_resume_source_unchanged(
            source_root=source,
            files=files,
            source_identities=identities,
            state=state,
        )


def test_resume_requires_the_exact_manifest_file_set(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    first = source / "classes.txt"
    first.parent.mkdir(parents=True)
    first.write_text("aircraft\n", encoding="utf-8")
    files = [{"relative_path": "classes.txt", "size_bytes": first.stat().st_size}]
    identities = _capture_manifest_identities(source, files)

    with pytest.raises(SourceFileChangedError, match="resume source file set changed"):
        _assert_resume_source_unchanged(
            source_root=source,
            files=files,
            source_identities=identities,
            state={"files": {}},
        )


def _write_invalid_detection_dataset(root: Path) -> None:
    image = root / "images" / "train" / "sample.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (16, 12)).save(image)
    label = root / "labels" / "train" / "sample.txt"
    label.parent.mkdir(parents=True)
    label.write_text("0 0.5 0.5 0.25\n", encoding="utf-8")
    (root / "classes.txt").write_text("aircraft\n", encoding="utf-8")


def test_folder_upload_validates_dataset_before_session_or_state_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_invalid_detection_dataset(tmp_path)
    state_dir = tmp_path / "state"
    args = _build_parser().parse_args(
        [
            "dataset",
            "upload",
            "--project",
            "proj_123456789abc",
            "--source",
            str(tmp_path),
            "--format",
            "yolo",
            "--task-key",
            "detect",
            "--concurrency",
            "1",
            "--batch-size",
            "1",
            "--hash-workers",
            "1",
            "--batch-complete-concurrency",
            "1",
            "--stream-flush-size",
            "1",
            "--state-dir",
            str(state_dir),
        ]
    )
    monkeypatch.setattr("avia_cli.core.uploads.dataset.probe_rtt_seconds", lambda *_args: 0.001)
    monkeypatch.setattr(
        "avia_cli.core.uploads.dataset._create_dataset_session",
        lambda **_kwargs: pytest.fail("invalid dataset reached HTTP session creation"),
    )

    with pytest.raises(SystemExit, match="invalid_yolo_detect_row"):
        upload_dataset(args, api="https://avia.eurekailab.com/api/v1", token="token")

    assert not state_dir.exists()


def test_folder_upload_persists_pending_uuid_session_before_first_post_and_resume_replays_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "dataset"
    image = source / "images" / "train" / "sample.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (16, 12)).save(image)
    label = source / "labels" / "train" / "sample.txt"
    label.parent.mkdir(parents=True)
    label.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    (source / "classes.txt").write_text("aircraft\n", encoding="utf-8")
    state_dir = tmp_path / "state"
    parser = _build_parser()
    argv = [
        "dataset",
        "upload",
        "--project",
        "proj_123456789abc",
        "--source",
        str(source),
        "--format",
        "yolo",
        "--task-key",
        "detect",
        "--concurrency",
        "1",
        "--batch-size",
        "1",
        "--hash-workers",
        "1",
        "--batch-complete-concurrency",
        "1",
        "--stream-flush-size",
        "1",
        "--state-dir",
        str(state_dir),
    ]
    seen_payloads: list[dict[str, object]] = []

    def lose_response(**kwargs: object) -> dict[str, object]:
        state_paths = list(state_dir.rglob("*.json"))
        assert len(state_paths) == 1, "pending session must be durable before the first POST"
        state = json.loads(state_paths[0].read_text(encoding="utf-8"))
        assert state["phase"] == "session_pending"
        assert state["import_id"] is None
        assert kwargs["payload"] == state["session_payload"]
        seen_payloads.append(dict(kwargs["payload"]))
        raise RuntimeError("response lost after server accepted session")

    monkeypatch.setattr("avia_cli.core.uploads.dataset._create_dataset_session", lose_response)

    with pytest.raises(RuntimeError, match="response lost"):
        upload_dataset(parser.parse_args(argv), api="https://avia.example/api/v1", token="token")

    first_payload = seen_payloads[0]
    key = str(first_payload["idempotency_key"])
    assert re.fullmatch(
        r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}",
        key,
    )
    assert set(first_payload) == {
        "idempotency_key",
        "format",
        "root_name",
        "task_key",
        "classes",
        "file_count",
        "total_bytes",
    }

    with pytest.raises(RuntimeError, match="response lost"):
        upload_dataset(
            parser.parse_args([*argv, "--resume"]),
            api="https://avia.example/api/v1",
            token="token",
        )

    assert seen_payloads == [first_payload, first_payload]


def test_anomalib_folder_upload_uses_binary_taxonomy_in_session_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "canonical-ad"
    for relative in (
        "train/good/train.png",
        "val/good/val-good.png",
        "val/bad/val-bad.png",
        "test/good/test-good.png",
        "test/bad/test-bad.png",
    ):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 12)).save(path)
    for relative in (
        "ground_truth/val/bad/val-bad.png",
        "ground_truth/test/bad/test-bad.png",
    ):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", (16, 12), color=255).save(path)

    argv = [
        "dataset",
        "upload",
        "--project",
        "proj_123456789abc",
        "--source",
        str(source),
        "--format",
        "anomalib",
        "--task-key",
        "ad",
        "--concurrency",
        "1",
        "--batch-size",
        "100",
        "--hash-workers",
        "1",
        "--batch-complete-concurrency",
        "1",
        "--stream-flush-size",
        "1",
        "--state-dir",
        str(tmp_path / "state"),
    ]
    parser = _build_parser()
    seen_payloads: list[dict[str, object]] = []

    def capture_payload(**kwargs: object) -> dict[str, object]:
        payload = kwargs.get("payload")
        assert isinstance(payload, dict)
        seen_payloads.append(dict(payload))
        raise RuntimeError("stop after session payload")

    monkeypatch.setattr("avia_cli.core.uploads.dataset._create_dataset_session", capture_payload)

    with pytest.raises(RuntimeError, match="stop after session payload"):
        upload_dataset(
            parser.parse_args(argv),
            api="https://avia.example/api/v1",
            token="token",
        )
    with pytest.raises(RuntimeError, match="stop after session payload"):
        upload_dataset(
            parser.parse_args([*argv, "--resume"]),
            api="https://avia.example/api/v1",
            token="token",
        )

    assert len(seen_payloads) == 2
    assert seen_payloads[0] == seen_payloads[1]
    assert seen_payloads[0]["format"] == "anomalib"
    assert seen_payloads[0]["task_key"] == "ad"
    assert seen_payloads[0]["classes"] == ["good", "bad"]


def test_folder_upload_waits_for_running_puts_and_persists_their_success_after_peer_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "dataset"
    for name in ("a", "b"):
        image = source / "images" / "train" / f"{name}.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 12)).save(image)
        label = source / "labels" / "train" / f"{name}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    (source / "classes.txt").write_text("aircraft\n", encoding="utf-8")
    state_dir = tmp_path / "state"
    args = _build_parser().parse_args(
        [
            "dataset",
            "upload",
            "--project",
            "proj_123456789abc",
            "--source",
            str(source),
            "--format",
            "yolo",
            "--task-key",
            "detect",
            "--concurrency",
            "8",
            "--batch-size",
            "100",
            "--hash-workers",
            "2",
            "--batch-complete-concurrency",
            "1",
            "--stream-flush-size",
            "100",
            "--state-dir",
            str(state_dir),
        ]
    )
    monkeypatch.setattr(
        "avia_cli.core.uploads.dataset._create_dataset_session",
        lambda **_kwargs: {"import_id": "imp_concurrent_failure"},
    )
    monkeypatch.setattr(
        "avia_cli.core.uploads.dataset._batch_upload_urls",
        lambda **kwargs: {
            "files": [
                {
                    "relative_path": item["relative_path"],
                    "upload_url": f"https://objects.example/{item['relative_path']}",
                    "object_key": f"objects/{item['relative_path']}",
                    "required_headers": {},
                }
                for item in kwargs["files"]
            ]
        },
    )
    success_started = threading.Event()
    failure_released = threading.Event()
    success_finished = threading.Event()
    probe_routes: list[object] = []
    put_routes: list[object] = []
    monkeypatch.setattr(
        "avia_cli.core.uploads.dataset._resolve_transport_concurrency",
        lambda _args, *, route: probe_routes.append(route),
    )

    def put_with_one_failure(**kwargs: object) -> None:
        put_routes.append(kwargs["route"])
        relative_path = Path(str(kwargs["path"])).relative_to(source).as_posix()
        if relative_path == "labels/train/a.txt":
            assert success_started.wait(timeout=1)
            failure_released.set()
            raise RuntimeError("one PUT failed")
        if relative_path == "labels/train/b.txt":
            success_started.set()
            assert failure_released.wait(timeout=1)
            time.sleep(0.05)
            success_finished.set()

    monkeypatch.setattr(
        "avia_cli.core.uploads.dataset._put_file_with_retries",
        put_with_one_failure,
    )

    with pytest.raises(RuntimeError, match="one PUT failed"):
        upload_dataset(args, api="https://avia.eurekailab.com/api/v1", token="token")

    assert success_finished.is_set(), "upload returned while a sibling PUT still mutated storage"
    assert len(probe_routes) == 1
    assert any(route is probe_routes[0] for route in put_routes)
    state_path = next(state_dir.rglob("*.json"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["files"]["labels/train/b.txt"]["uploaded"] is True


def test_folder_upload_drains_stream_completions_and_persists_success_after_peer_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "dataset"
    for name in ("a", "b"):
        image = source / "images" / "train" / f"{name}.png"
        image.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (16, 12)).save(image)
        label = source / "labels" / "train" / f"{name}.txt"
        label.parent.mkdir(parents=True, exist_ok=True)
        label.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    (source / "classes.txt").write_text("aircraft\n", encoding="utf-8")
    state_dir = tmp_path / "state"
    args = _build_parser().parse_args(
        [
            "dataset",
            "upload",
            "--project",
            "proj_123456789abc",
            "--source",
            str(source),
            "--format",
            "yolo",
            "--task-key",
            "detect",
            "--concurrency",
            "8",
            "--batch-size",
            "100",
            "--hash-workers",
            "2",
            "--batch-complete-concurrency",
            "8",
            "--stream-flush-size",
            "1",
            "--state-dir",
            str(state_dir),
        ]
    )
    monkeypatch.setattr(
        "avia_cli.core.uploads.dataset._create_dataset_session",
        lambda **_kwargs: {"import_id": "imp_stream_failure"},
    )
    monkeypatch.setattr(
        "avia_cli.core.uploads.dataset._batch_upload_urls",
        lambda **kwargs: {
            "files": [
                {
                    "relative_path": item["relative_path"],
                    "upload_url": f"https://objects.example/{item['relative_path']}",
                    "object_key": f"objects/{item['relative_path']}",
                    "required_headers": {},
                }
                for item in kwargs["files"]
            ]
        },
    )
    monkeypatch.setattr(
        "avia_cli.core.uploads.dataset._put_file_with_retries",
        lambda **_kwargs: None,
    )
    success_started = threading.Event()
    failure_released = threading.Event()
    success_finished = threading.Event()

    def complete_with_one_failure(**kwargs: object) -> dict[str, object]:
        relative_path = str(kwargs["files"][0]["relative_path"])
        if relative_path == "labels/train/a.txt":
            assert success_started.wait(timeout=1)
            failure_released.set()
            raise RuntimeError("one stream completion failed")
        if relative_path == "labels/train/b.txt":
            success_started.set()
            assert failure_released.wait(timeout=1)
            time.sleep(0.05)
            success_finished.set()
        return {"accepted": [relative_path]}

    monkeypatch.setattr(
        "avia_cli.core.uploads.dataset._complete_dataset_file_batch",
        complete_with_one_failure,
    )

    with pytest.raises(RuntimeError, match="one stream completion failed"):
        upload_dataset(args, api="https://avia.eurekailab.com/api/v1", token="token")

    assert success_finished.is_set(), "upload returned while stream completion still ran"
    state_path = next(state_dir.rglob("*.json"))
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["files"]["labels/train/b.txt"]["streamed"] is True


def test_folder_upload_validates_dataset_before_auto_tuning_network_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_invalid_detection_dataset(tmp_path)
    args = _build_parser().parse_args(
        [
            "dataset",
            "upload",
            "--project",
            "proj_123abc456def",
            "--source",
            str(tmp_path),
            "--format",
            "yolo",
            "--task-key",
            "detect",
        ]
    )
    monkeypatch.setattr(
        "avia_cli.core.uploads.dataset.probe_rtt_seconds",
        lambda *_args, **_kwargs: pytest.fail("invalid dataset reached the network RTT probe"),
    )
    monkeypatch.setattr(
        "avia_cli.core.uploads.dataset._create_dataset_session",
        lambda **_kwargs: pytest.fail("invalid dataset reached the Avia API"),
    )

    with pytest.raises(SystemExit, match="invalid_yolo_detect_row"):
        upload_dataset(args, api="https://avia.example/api/v1", token="token")


def test_folder_upload_rejects_class_override_for_non_yolo_before_scanning(
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
        "avia_cli.core.uploads.dataset.scan_source_manifest",
        lambda *_args, **_kwargs: pytest.fail("invalid --class scope reached dataset scan"),
    )

    with pytest.raises(SystemExit, match="--class is only valid with --format yolo"):
        upload_dataset(args, api="https://avia.example/api/v1", token="token")


def test_folder_upload_emits_and_returns_segment_topology_warnings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    image = tmp_path / "images" / "train" / "sample.png"
    image.parent.mkdir(parents=True)
    Image.new("RGB", (16, 12)).save(image)
    label = tmp_path / "labels" / "train" / "sample.txt"
    label.parent.mkdir(parents=True)
    label.write_text("0 0.1 0.1 0.9 0.9 0.1 0.8 0.8 0.1\n", encoding="utf-8")
    (tmp_path / "classes.txt").write_text("aircraft\n", encoding="utf-8")
    args = _build_parser().parse_args(
        [
            "dataset",
            "upload",
            "--project",
            "proj_123abc456def",
            "--source",
            str(tmp_path),
            "--format",
            "yolo",
            "--task-key",
            "segment",
            "--concurrency",
            "1",
            "--batch-size",
            "1",
            "--hash-workers",
            "1",
            "--batch-complete-concurrency",
            "1",
            "--stream-flush-size",
            "1",
            "--state-dir",
            str(tmp_path / "state"),
        ]
    )
    monkeypatch.setattr(
        "avia_cli.core.uploads.dataset._upload_validated_dataset",
        lambda **_kwargs: {"status": "accepted"},
    )

    result = upload_dataset(args, api="https://avia.example/api/v1", token="token")

    assert result["validation_warnings"][0]["code"] == "yolo_segment_topology"
    event = json.loads(capsys.readouterr().err)
    assert event["event"] == "dataset_validation_warnings"
    assert event["warnings"] == result["validation_warnings"]


def test_archive_upload_command_is_removed() -> None:
    with pytest.raises(SystemExit):
        _build_parser().parse_args(["dataset", "upload-archive"])
