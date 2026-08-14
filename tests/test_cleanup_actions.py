from __future__ import annotations

from copy import deepcopy

import pytest

from avia_cli.core.uploads.inspect import _build_cleanup_actions


def _local_state() -> dict[str, object]:
    return {
        "project_id": "proj_1",
        "import_id": "imp_done",
        "source": "/data/coco8",
        "format": "yolo",
        "task_key": "detect",
        "phase": "completed",
        "total_files": 1,
        "uploaded_files": 1,
        "streamed_files": 1,
        "state_path": "/data/coco8/.avia/imports/imp_done.json",
    }


def _server_import(*, status: object = "succeeded") -> dict[str, object]:
    return {
        "import_id": "imp_done",
        "status": status,
        "job_type": "dataset.import.yolo",
        "object_key": "project_assets/ws_1/scope_1/imports/imp_done/manifest.json",
    }


def test_cleanup_action_requires_exact_canonical_identity() -> None:
    assert _build_cleanup_actions(
        local_states=[_local_state()],
        server_imports=[_server_import()],
    ) == [
        {
            "kind": "remove_local_state",
            "path": "/data/coco8/.avia/imports/imp_done.json",
            "reason": "server import is terminal and local resume state is completed",
        }
    ]


@pytest.mark.parametrize(
    ("target", "field", "value"),
    [
        ("server", "status", "SUCCEEDED"),
        ("server", "import_id", 7),
        ("server", "legacy", "value"),
        ("local", "phase", "COMPLETED"),
        ("local", "import_id", 7),
        ("local", "legacy", "value"),
    ],
)
def test_cleanup_action_rejects_coercion_aliases_and_unknown_fields(
    target: str,
    field: str,
    value: object,
) -> None:
    local = _local_state()
    server = _server_import()
    (local if target == "local" else server)[field] = value

    with pytest.raises(RuntimeError):
        _build_cleanup_actions(local_states=[local], server_imports=[server])


def test_cleanup_action_rejects_duplicate_server_imports_independent_of_order() -> None:
    terminal = _server_import(status="succeeded")
    running = _server_import(status="running")
    for entries in ([terminal, running], [running, terminal]):
        with pytest.raises(RuntimeError, match="duplicate imp_done"):
            _build_cleanup_actions(
                local_states=[_local_state()],
                server_imports=deepcopy(entries),
            )
