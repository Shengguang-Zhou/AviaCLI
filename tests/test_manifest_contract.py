from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from avia_cli.core.uploads.manifest import scan_source_manifest


def test_manifest_rejects_fifo_instead_of_silently_skipping_it(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    source.mkdir()
    os.mkfifo(source / "image-stream")

    with pytest.raises(SystemExit) as exc_info:
        scan_source_manifest(source, format_name="anomalib", include_dimensions=False)

    assert json.loads(str(exc_info.value)) == {
        "code": "unsupported_dataset_member",
        "message": "dataset source tree must contain only regular files and directories",
        "path": "image-stream",
    }
