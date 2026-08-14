from __future__ import annotations

import threading
from copy import deepcopy
from functools import partial
from pathlib import Path

import pytest

from avia_cli.core.uploads.dataset import _PutStateTracker, _record_successful_put
from avia_cli.core.uploads.put_batch import PutSuccess, settle_concurrent_puts


def test_state_record_failure_cannot_leave_running_puts_unsettled() -> None:
    release_sibling = threading.Event()
    sibling_started = threading.Event()
    sibling_finished = threading.Event()
    recorded: list[str] = []

    def upload_one(relative_path: str) -> PutSuccess:
        if relative_path == "a.txt":
            assert sibling_started.wait(timeout=1)
        else:
            sibling_started.set()
            assert release_sibling.wait(timeout=1)
            sibling_finished.set()
        return PutSuccess(
            relative_path=relative_path,
            signed={"object_key": f"objects/{relative_path}"},
            version_id=f"version-{relative_path}",
        )

    def record_success(relative_path: str, _signed: dict[str, object], _version_id: str) -> None:
        recorded.append(relative_path)
        if relative_path == "a.txt":
            release_sibling.set()
            raise OSError("resume state write failed")

    failures = settle_concurrent_puts(
        relative_paths=["a.txt", "b.txt"],
        max_workers=2,
        upload_one=upload_one,
        record_success=record_success,
    )

    assert sibling_finished.is_set()
    assert set(recorded) == {"a.txt", "b.txt"}
    assert [(phase, targets, str(error)) for phase, targets, error in failures] == [
        ("file-put-state", ["a.txt"], "resume state write failed")
    ]


def test_final_state_save_contains_every_success_after_periodic_save_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_sibling = threading.Event()
    sibling_started = threading.Event()
    final_snapshots: list[dict[str, object]] = []
    save_attempts = 0
    state_files = {
        path: {
            "uploaded": False,
            "streamed": False,
            "object_key": None,
            "version_id": None,
            "content_type": None,
            "sha256": "",
            "size_bytes": 1,
            "width": 0,
            "height": 0,
            "source_identity": {},
        }
        for path in ("a.txt", "b.txt")
    }
    state: dict[str, object] = {"files": state_files}
    tracker = _PutStateTracker(
        ready_to_stream=[],
        progress={
            "started_at": 1.0,
            "total_bytes": 2,
            "done_bytes": 0,
            "done_files": 0,
            "total_files": 2,
            "last_at": 0.0,
        },
        pending_state_saves=0,
        last_state_saved_at=0.0,
    )

    def save_state(_state_dir: Path, current: dict[str, object]) -> None:
        nonlocal save_attempts
        save_attempts += 1
        if save_attempts == 1:
            release_sibling.set()
            raise OSError("resume state write failed")
        final_snapshots.append(deepcopy(current))

    monkeypatch.setattr("avia_cli.core.uploads.dataset._save_state", save_state)

    def upload_one(relative_path: str) -> PutSuccess:
        if relative_path == "a.txt":
            assert sibling_started.wait(timeout=1)
        else:
            sibling_started.set()
            assert release_sibling.wait(timeout=1)
        return PutSuccess(
            relative_path=relative_path,
            signed={
                "object_key": f"objects/{relative_path}",
                "content_type": "text/plain",
            },
            version_id=f"version-{relative_path}",
        )

    failures = settle_concurrent_puts(
        relative_paths=["a.txt", "b.txt"],
        max_workers=2,
        upload_one=upload_one,
        record_success=partial(
            _record_successful_put,
            file_by_relative={
                "a.txt": {"sha256": "a" * 64, "size_bytes": 1},
                "b.txt": {"sha256": "b" * 64, "size_bytes": 1},
            },
            state_files=state_files,
            state=state,
            state_dir=tmp_path,
            tracker=tracker,
            state_flush_every=1,
            state_flush_interval=60.0,
            emit_progress=lambda: None,
        ),
    )
    save_state(tmp_path, state)

    assert [(phase, targets, str(error)) for phase, targets, error in failures] == [
        ("file-put-state", ["a.txt"], "resume state write failed")
    ]
    assert set(tracker.ready_to_stream) == {"a.txt", "b.txt"}
    persisted_files = final_snapshots[-1]["files"]
    assert isinstance(persisted_files, dict)
    assert persisted_files["a.txt"]["uploaded"] is True
    assert persisted_files["b.txt"]["uploaded"] is True
    assert persisted_files["a.txt"]["version_id"] == "version-a.txt"
    assert persisted_files["b.txt"]["version_id"] == "version-b.txt"


def test_out_of_order_puts_keep_each_receipt_bound_to_its_path() -> None:
    release_a = threading.Event()
    b_finished = threading.Event()
    recorded: dict[str, str] = {}

    def upload_one(relative_path: str) -> PutSuccess:
        if relative_path == "a.txt":
            assert release_a.wait(timeout=1)
        else:
            b_finished.set()
            release_a.set()
        return PutSuccess(
            relative_path=relative_path,
            signed={"object_key": f"objects/{relative_path}"},
            version_id=f"version-{relative_path}",
        )

    failures = settle_concurrent_puts(
        relative_paths=["a.txt", "b.txt"],
        max_workers=2,
        upload_one=upload_one,
        record_success=lambda path, _signed, version_id: recorded.__setitem__(path, version_id),
    )

    assert b_finished.is_set()
    assert failures == []
    assert recorded == {
        "a.txt": "version-a.txt",
        "b.txt": "version-b.txt",
    }


def test_telemetry_failure_is_reported_after_remote_success_is_recorded() -> None:
    recorded: list[str] = []
    telemetry_error = OSError("timing sink failed")

    failures = settle_concurrent_puts(
        relative_paths=["a.txt"],
        max_workers=1,
        upload_one=lambda relative_path: PutSuccess(
            relative_path=relative_path,
            signed={"object_key": "objects/a.txt"},
            version_id="version-a.txt",
            telemetry_error=telemetry_error,
        ),
        record_success=lambda relative_path, _signed, _version_id: recorded.append(relative_path),
    )

    assert recorded == ["a.txt"]
    assert failures == [("file-put-telemetry", ["a.txt"], telemetry_error)]
