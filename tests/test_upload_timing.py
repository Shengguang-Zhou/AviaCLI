from __future__ import annotations

import pytest

from avia_cli.core.uploads.timing import (
    UploadTimingRecorder,
    UploadTimingRecordError,
)


def test_capture_call_preserves_success_when_timing_record_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = UploadTimingRecorder()
    side_effects: list[str] = []

    def fail_record(*_args: object, **_kwargs: object) -> None:
        raise OSError("timing sink failed")

    monkeypatch.setattr(recorder, "record", fail_record)

    outcome = recorder.capture_call(
        "remote-put",
        lambda: side_effects.append("stored") or "response",
    )

    assert side_effects == ["stored"]
    assert outcome.value == "response"
    assert isinstance(outcome.telemetry_error, UploadTimingRecordError)
    assert isinstance(outcome.telemetry_error.record_error, OSError)


def test_time_call_reports_operation_and_timing_failures_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder = UploadTimingRecorder()
    operation_error = ValueError("remote failed")

    def fail_record(*_args: object, **_kwargs: object) -> None:
        raise OSError("timing sink failed")

    def fail_operation() -> None:
        raise operation_error

    monkeypatch.setattr(recorder, "record", fail_record)

    with pytest.raises(UploadTimingRecordError) as captured:
        recorder.time_call("remote-call", fail_operation)

    assert captured.value.operation_error is operation_error
    assert isinstance(captured.value.record_error, OSError)
