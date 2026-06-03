from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from avia_cli.core.uploads.refs import attach_upload_refs
from avia_cli.core.uploads.manifest import scan_source_manifest
from avia_cli.core.uploads.urls import upload_request_from_api


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


def test_scan_source_manifest_reads_yolo_images_and_labels(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    images = source / "images" / "train"
    labels = source / "labels" / "train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    image = images / "a.jpg"
    label = labels / "a.txt"
    image.write_bytes(b"not-a-real-image")
    label.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")

    manifest = scan_source_manifest(source, format_name="yolo", include_dimensions=False)

    paths = {str(item["relative_path"]) for item in manifest["files"]}
    assert "images/train/a.jpg" in paths
    assert "labels/train/a.txt" in paths
    assert manifest["file_count"] == 2


def test_upload_request_from_api_rewrites_public_upload_url() -> None:
    url, headers = upload_request_from_api(
        api="https://avia.eurekailab.com/api/v1",
        raw_url="https://avia.eurekailab.com/avia-runtime/project_assets/a.jpg?X-Amz-Signature=1",
        upload_url_origin_override="http://127.0.0.1:9000",
    )

    assert url == (
        "http://127.0.0.1:9000/avia-runtime/project_assets/a.jpg?X-Amz-Signature=1"
    )
    assert headers == {"Host": "avia.eurekailab.com"}


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


def test_attach_upload_refs_promotes_archive_artifact_ref() -> None:
    result = {
        "multipart": {
            "complete": {
                "artifact_ref": {
                    "id": "ar_import",
                    "storage": {"kind": "minio", "object_key": "raw.zip"},
                },
                "read_lease": {
                    "id": "lease_archive",
                    "artifact_ref_id": "ar_import",
                },
            }
        },
        "complete": {},
    }

    attached = attach_upload_refs(result)

    assert attached["artifact_ref"]["id"] == "ar_import"
    assert attached["read_lease"]["artifact_ref_id"] == "ar_import"
