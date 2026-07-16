from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from avia_cli.core.uploads.refs import attach_upload_refs
from avia_cli.core.uploads.manifest import scan_source_manifest
from avia_cli.parser import _build_parser


def test_module_entrypoint_prints_help() -> None:
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-m", "avia_cli.main", "--help"],
        text=True,
        capture_output=True,
        check=False,
        env={"PYTHONPATH": str(root / "packages" / "avia-cli" / "src")},
    )

    assert proc.returncode == 0
    assert "usage: avia" in proc.stdout
    assert "{auth,import,dataset}" in proc.stdout


def test_dataset_help_documents_exact_format_task_matrix() -> None:
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-m", "avia_cli.main", "dataset", "verify", "--help"],
        text=True,
        capture_output=True,
        check=False,
        env={"PYTHONPATH": str(root / "packages" / "avia-cli" / "src")},
    )

    assert proc.returncode == 0
    assert "--format" in proc.stdout
    assert "--task-key" in proc.stdout
    assert "yolo=detect,classify,segment,pose,obb" in proc.stdout
    assert "coco=detect,segment,pose" in proc.stdout
    assert "imagenet=classify" in proc.stdout
    assert "anomalib=ad" in proc.stdout


def test_scan_source_manifest_reads_yolo_images_and_labels(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    images = source / "images" / "train"
    labels = source / "labels" / "train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    image = images / "a.jpg"
    label = labels / "a.txt"
    from PIL import Image

    Image.new("RGB", (16, 12)).save(image)
    label.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")

    manifest = scan_source_manifest(source, format_name="yolo", include_dimensions=False)

    paths = {str(item["relative_path"]) for item in manifest["files"]}
    assert "images/train/a.jpg" in paths
    assert "labels/train/a.txt" in paths
    assert manifest["file_count"] == 2


def test_upload_origin_override_benchmark_bypass_is_not_a_cli_option() -> None:
    with pytest.raises(SystemExit):
        _build_parser().parse_args(
            [
                "dataset",
                "upload",
                "--project",
                "proj_123456789abc",
                "--source",
                "/data/example",
                "--format",
                "yolo",
                "--task-key",
                "detect",
                "--upload-url-origin-override",
                "http://127.0.0.1:9000",
            ]
        )


def test_attach_upload_refs_promotes_dataset_manifest_ref() -> None:
    result = {
        "complete": {
            "dataset_manifest_ref": {
                "id": "dm_import",
                "storage": {"kind": "minio", "manifest_path": "manifest.json"},
            },
            "read_lease": {
                "id": "lease_import",
                "dataset_manifest_ref_id": "dm_import",
            },
        }
    }

    attached = attach_upload_refs(result)

    assert attached["dataset_manifest_ref"]["id"] == "dm_import"
    assert attached["read_lease"]["id"] == "lease_import"
