from __future__ import annotations

import json

import pytest

from avia_cli.commands.dataset import _print_upload_result
from avia_cli.core.errors import decode_json_response
from avia_cli.core.strict_json import strict_json_loads
from avia_cli.core.uploads.contracts import require_format_task, require_object_prefix_uri
from avia_cli.core.uploads.state import _require_idempotency_key


@pytest.mark.parametrize(
    "document",
    [
        '{"status":"failed","status":"queued"}',
        '{"outer":{"status":"failed","status":"queued"}}',
        '{"status":NaN}',
        '{"status":Infinity}',
        '{"status":-Infinity}',
        '{"score":1e400}',
    ],
)
def test_strict_json_rejects_ambiguous_or_nonfinite_documents(document: str) -> None:
    with pytest.raises(ValueError):
        strict_json_loads(document)


@pytest.mark.parametrize(
    "document",
    [
        b'{"status":"failed","status":"queued"}',
        b'{"status":NaN}',
    ],
)
def test_http_json_decoder_rejects_ambiguous_wire_payloads(document: bytes) -> None:
    with pytest.raises(RuntimeError) as captured:
        decode_json_response(document, url="https://avia.example/api/v1/imports/imp_1")

    payload = json.loads(str(captured.value))
    assert payload["code"] == "invalid_json_response"
    assert payload["message"] == "HTTP response violates the strict JSON contract"


class _StringLike:
    def __str__(self) -> str:
        return "yolo"


def test_upload_contracts_reject_string_coercion_at_direct_boundaries() -> None:
    with pytest.raises(SystemExit, match="format and task must be strings"):
        require_format_task(format_name=_StringLike(), task_key="detect")
    with pytest.raises(SystemExit, match="URI must be a string"):
        require_object_prefix_uri(_StringLike())
    with pytest.raises(ValueError, match="exact string"):
        _require_idempotency_key(_StringLike())


@pytest.mark.parametrize(
    "result",
    [
        {"file_count": 1, "project_id": "proj_1", "import_id": "imp_1"},
        {
            "file_count": 1,
            "project_id": "proj_1",
            "import_id": "imp_1",
            "complete": {},
        },
        {
            "file_count": 1,
            "project_id": "proj_1",
            "import_id": "imp_1",
            "complete": {"status": "QUEUED"},
        },
    ],
)
def test_human_upload_output_never_fabricates_or_normalizes_status(
    result: dict[str, object],
) -> None:
    with pytest.raises(RuntimeError):
        _print_upload_result(result, json_output=False)
