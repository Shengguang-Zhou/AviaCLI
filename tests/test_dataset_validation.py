from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from PIL import Image
from pycocotools import mask as mask_utils

from avia_cli.core.uploads.inspect import verify_dataset
from avia_cli.core.uploads.manifest import scan_source_manifest
from avia_cli.core.uploads.validation import require_valid_dataset


def _write_image(path: Path, *, size: tuple[int, int] = (16, 12)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color=(20, 40, 60)).save(path)


def _write_mask(path: Path, *, size: tuple[int, int] = (16, 12)) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("L", size, color=255).save(path)


def _write_yolo_dataset(
    root: Path,
    *,
    label: str,
    names: list[str] | None = None,
    kpt_shape: list[int] | None = None,
) -> None:
    _write_image(root / "images" / "train" / "sample.png")
    label_path = root / "labels" / "train" / "sample.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text(label, encoding="utf-8")
    metadata: dict[str, object] = {"names": names if names is not None else ["aircraft"]}
    if kpt_shape is not None:
        metadata["kpt_shape"] = kpt_shape
    (root / "data.yaml").write_text(yaml.safe_dump(metadata), encoding="utf-8")


def _verify_yolo(root: Path, task_key: str) -> dict[str, object]:
    return verify_dataset(source=root, format_name="yolo", task_key=task_key)


@pytest.mark.parametrize(
    ("format_name", "task_key"),
    [("YOLO", "detect"), ("yolo", "Detect"), (" yolo", "detect")],
)
def test_verify_rejects_noncanonical_format_or_task_aliases(
    tmp_path: Path, format_name: str, task_key: str
) -> None:
    with pytest.raises(SystemExit, match="does not support"):
        verify_dataset(source=tmp_path, format_name=format_name, task_key=task_key)


@pytest.mark.parametrize(
    ("task_key", "label"),
    [
        ("detect", "0 0.5 0.5 0.25 0.25\n"),
        ("classify", "0\n"),
        ("segment", "0 0.1 0.1 0.8 0.1 0.5 0.8\n"),
        ("obb", "0 0.1 0.1 0.8 0.1 0.8 0.8 0.1 0.8\n"),
    ],
)
def test_verify_yolo_accepts_exact_valid_task_rows(
    tmp_path: Path, task_key: str, label: str
) -> None:
    _write_yolo_dataset(tmp_path, label=label)

    result = _verify_yolo(tmp_path, task_key)

    assert result["status"] == "ok"
    assert result["task_key"] == task_key
    assert result["error_count"] == 0
    assert result["warning_count"] == 0


def test_verify_yolo_pose_uses_exact_kpt_shape(tmp_path: Path) -> None:
    _write_yolo_dataset(
        tmp_path,
        label="0 0.5 0.5 0.25 0.25 0.2 0.3 2 0.7 0.8 1\n",
        kpt_shape=[2, 3],
    )

    result = _verify_yolo(tmp_path, "pose")

    assert result["status"] == "ok"
    assert result["error_count"] == 0


def test_verify_yolo_pose_accepts_six_decimal_boundary_rounding(tmp_path: Path) -> None:
    _write_yolo_dataset(
        tmp_path,
        label="0 0.662641 0.494385 0.674719 0.988771 0.5 0.5 2\n",
        kpt_shape=[1, 3],
    )

    result = _verify_yolo(tmp_path, "pose")

    assert result["status"] == "ok"


def test_verify_yolo_segment_reports_nonzero_self_intersection_as_warning(tmp_path: Path) -> None:
    _write_yolo_dataset(
        tmp_path,
        label="0 0.1 0.1 0.9 0.9 0.1 0.8 0.8 0.1\n",
    )

    result = _verify_yolo(tmp_path, "segment")

    assert result["status"] == "ok"
    assert result["errors"] == []
    assert result["warnings"] == [
        {
            "code": "yolo_segment_topology",
            "path": "labels/train/sample.txt",
            "line": 1,
            "message": "segment is rasterizable but contains crossing or non-canonical bridge topology",
        }
    ]


def test_verify_yolo_segment_accepts_real_ultralytics_thin_bridge_multisegment(
    tmp_path: Path,
) -> None:
    official = Path("tests/fixtures/ultralytics_coco8_seg_bridge.txt").read_text(encoding="utf-8")
    _class_id, coordinates = official.split(maxsplit=1)
    _write_yolo_dataset(tmp_path, label=f"0 {coordinates}")

    result = _verify_yolo(tmp_path, "segment")

    assert result["status"] == "ok"


def test_verify_yolo_segment_accepts_weak_simple_reverse_overlap_and_duplicate_point(
    tmp_path: Path,
) -> None:
    _write_yolo_dataset(
        tmp_path,
        label="0 0.1 0.1 0.3 0.1 0.5 0.3 0.3 0.5 0.2 0.1 0.2 0.1\n",
    )

    result = _verify_yolo(tmp_path, "segment")

    assert result["status"] == "ok"


def test_verify_yolo_segment_reports_same_direction_overlap_as_warning(tmp_path: Path) -> None:
    _write_yolo_dataset(
        tmp_path,
        label="0 0.1 0.1 0.9 0.1 0.9 0.9 0.3 0.1 0.7 0.1 0.1 0.9\n",
    )

    result = _verify_yolo(tmp_path, "segment")

    assert result["status"] == "ok"
    assert result["warning_count"] == 1
    assert result["warnings"][0]["code"] == "yolo_segment_topology"


@pytest.mark.parametrize(
    ("task_key", "label"),
    [
        ("detect", "0 0.5 0.5 0.25\n"),
        ("detect", "0.0 0.5 0.5 0.25 0.25\n"),
        ("detect", "0 nan 0.5 0.25 0.25\n"),
        ("detect", "0 1.1 0.5 0.25 0.25\n"),
        ("detect", "0 0.5 0.5 0 0.25\n"),
        ("classify", "0\n0\n"),
        ("segment", "0 0.1 0.1 0.8 0.1\n"),
        ("segment", "0 0.1 0.1 0.2 0.2 0.3 0.3\n"),
        ("obb", "0 0.1 0.1 0.8 0.8 0.8 0.1 0.1 0.8\n"),
        ("obb", "0 0.1 0.1 0.9 0.1 0.5 0.5 0.1 0.9\n"),
    ],
)
def test_verify_yolo_rejects_invalid_task_rows(tmp_path: Path, task_key: str, label: str) -> None:
    _write_yolo_dataset(tmp_path, label=label)

    result = _verify_yolo(tmp_path, task_key)

    assert result["status"] == "failed"
    assert result["error_count"] >= 1


def test_verify_yolo_classify_accepts_existing_empty_label_as_explicit_negative(
    tmp_path: Path,
) -> None:
    _write_yolo_dataset(tmp_path, label="", names=["person", "potted plant"])

    result = _verify_yolo(tmp_path, "classify")

    assert result["status"] == "ok"
    assert result["error_count"] == 0


def test_verify_yolo_rejects_class_id_outside_declared_names(tmp_path: Path) -> None:
    _write_yolo_dataset(tmp_path, label="1 0.5 0.5 0.25 0.25\n", names=["only"])

    result = _verify_yolo(tmp_path, "detect")

    assert result["status"] == "failed"
    assert result["errors"][0]["code"] == "unknown_yolo_class"


def test_verify_yolo_rejects_missing_class_names(tmp_path: Path) -> None:
    _write_yolo_dataset(tmp_path, label="0 0.5 0.5 0.25 0.25\n")
    (tmp_path / "data.yaml").write_text("train: images/train\n", encoding="utf-8")

    result = _verify_yolo(tmp_path, "detect")

    assert result["status"] == "failed"
    assert result["errors"][0]["code"] == "missing_class_names"


@pytest.mark.parametrize(
    "names",
    [["aircraft", ""], ["aircraft", "aircraft"], [" aircraft"], [1]],
)
def test_verify_yolo_rejects_ambiguous_class_names(tmp_path: Path, names: list[object]) -> None:
    _write_yolo_dataset(tmp_path, label="0 0.5 0.5 0.25 0.25\n")
    (tmp_path / "data.yaml").write_text(yaml.safe_dump({"names": names}), encoding="utf-8")

    with pytest.raises(SystemExit, match="YOLO names"):
        _verify_yolo(tmp_path, "detect")


def test_verify_yolo_rejects_noncanonical_classes_txt(tmp_path: Path) -> None:
    _write_yolo_dataset(tmp_path, label="0 0.5 0.5 0.25 0.25\n")
    (tmp_path / "data.yaml").unlink()
    (tmp_path / "classes.txt").write_text("aircraft\n\nhelicopter \n", encoding="utf-8")

    with pytest.raises(SystemExit, match="canonical non-empty"):
        _verify_yolo(tmp_path, "detect")


@pytest.mark.parametrize("declared_classes", [[""], ["aircraft", "aircraft"]])
def test_upload_validation_rejects_invalid_declared_class_names(
    tmp_path: Path, declared_classes: list[str]
) -> None:
    _write_yolo_dataset(tmp_path, label="0 0.5 0.5 0.25 0.25\n")
    manifest = scan_source_manifest(tmp_path, format_name="yolo", include_dimensions=False)

    with pytest.raises(SystemExit, match="invalid_declared_class_names"):
        require_valid_dataset(
            source_root=tmp_path,
            manifest=manifest,
            format_name="yolo",
            task_key="detect",
            declared_classes=declared_classes,
        )


def test_upload_validation_rejects_conflicting_metadata_and_declared_classes(
    tmp_path: Path,
) -> None:
    _write_yolo_dataset(tmp_path, label="0 0.5 0.5 0.25 0.25\n")
    manifest = scan_source_manifest(tmp_path, format_name="yolo", include_dimensions=False)

    with pytest.raises(SystemExit, match="conflicting_class_names"):
        require_valid_dataset(
            source_root=tmp_path,
            manifest=manifest,
            format_name="yolo",
            task_key="detect",
            declared_classes=["plane"],
        )


def test_verify_yolo_rejects_missing_label_instead_of_warning(tmp_path: Path) -> None:
    _write_yolo_dataset(tmp_path, label="0 0.5 0.5 0.25 0.25\n")
    (tmp_path / "labels" / "train" / "sample.txt").unlink()

    result = _verify_yolo(tmp_path, "detect")

    assert result["status"] == "failed"
    assert result["errors"][0]["code"] == "missing_yolo_label"
    assert result["warning_count"] == 0


def test_verify_yolo_fully_decodes_every_image(tmp_path: Path) -> None:
    _write_yolo_dataset(tmp_path, label="0 0.5 0.5 0.25 0.25\n")
    image_path = tmp_path / "images" / "train" / "sample.png"
    image_path.write_bytes(image_path.read_bytes()[:-8])

    result = _verify_yolo(tmp_path, "detect")

    assert result["status"] == "failed"
    assert any(
        item["code"] == "invalid_image" and item["path"] == "images/train/sample.png"
        for item in result["errors"]
    )


def test_verify_yolo_compares_decoded_dimensions_with_manifest(tmp_path: Path) -> None:
    _write_yolo_dataset(tmp_path, label="0 0.5 0.5 0.25 0.25\n")
    manifest = scan_source_manifest(tmp_path, format_name="yolo")
    image_item = next(
        item for item in manifest["files"] if item["relative_path"] == "images/train/sample.png"
    )
    image_item["width"] = 99

    with pytest.raises(SystemExit, match="yolo_image_size_mismatch"):
        require_valid_dataset(
            source_root=tmp_path,
            manifest=manifest,
            format_name="yolo",
            task_key="detect",
        )


@pytest.mark.parametrize("kind", ["file", "directory", "broken"])
def test_manifest_rejects_every_source_symlink(tmp_path: Path, kind: str) -> None:
    source = tmp_path / "dataset"
    source.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    if kind == "file":
        target = outside / "secret.txt"
        target.write_text("secret", encoding="utf-8")
        (source / "link.txt").symlink_to(target)
    elif kind == "directory":
        (outside / "secret.txt").write_text("secret", encoding="utf-8")
        (source / "link-dir").symlink_to(outside, target_is_directory=True)
    else:
        (source / "broken.txt").symlink_to(outside / "missing.txt")

    with pytest.raises(SystemExit, match='"code": "dataset_symlink"'):
        scan_source_manifest(source, format_name="yolo", include_dimensions=False)


def test_manifest_rejects_control_characters_in_file_names(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    source.mkdir()
    (source / "bad\nname.txt").write_text("bad", encoding="utf-8")

    with pytest.raises(SystemExit, match='"code": "invalid_dataset_path"'):
        scan_source_manifest(source, format_name="yolo", include_dimensions=False)


def test_manifest_rejects_control_characters_in_source_root_name(tmp_path: Path) -> None:
    source = tmp_path / "bad\nroot"
    source.mkdir()

    with pytest.raises(SystemExit, match='"code": "invalid_dataset_path"'):
        scan_source_manifest(source, format_name="yolo", include_dimensions=False)


@pytest.mark.parametrize(
    ("kpt_shape", "label"),
    [
        (None, "0 0.5 0.5 0.25 0.25 0.2 0.3 2\n"),
        ([2, 3], "0 0.5 0.5 0.25 0.25 0.2 0.3 2\n"),
        ([1, 3], "0 0.5 0.5 0.25 0.25 0.2 0.3 1.5\n"),
        ([1, 2], "0 0.5 0.5 0.25 0.25 1.2 0.3\n"),
    ],
)
def test_verify_yolo_rejects_bad_pose_metadata_or_row(
    tmp_path: Path, kpt_shape: list[int] | None, label: str
) -> None:
    _write_yolo_dataset(tmp_path, label=label, kpt_shape=kpt_shape)

    result = _verify_yolo(tmp_path, "pose")

    assert result["status"] == "failed"


def test_verify_yolo_pose_requires_schema_even_when_label_is_empty(tmp_path: Path) -> None:
    _write_yolo_dataset(tmp_path, label="")

    result = _verify_yolo(tmp_path, "pose")

    assert any(item["code"] == "invalid_yolo_pose_metadata" for item in result["errors"])


def test_verify_yolo_rejects_box_whose_corners_leave_image(tmp_path: Path) -> None:
    _write_yolo_dataset(tmp_path, label="0 0.95 0.5 0.4 0.2\n")

    result = _verify_yolo(tmp_path, "detect")

    assert any(item["code"] == "invalid_yolo_detect_row" for item in result["errors"])


def test_verify_yolo_rejects_multiple_metadata_sources(tmp_path: Path) -> None:
    _write_yolo_dataset(tmp_path, label="0 0.5 0.5 0.2 0.2\n")
    (tmp_path / "classes.txt").write_text("aircraft\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="exactly one source of truth"):
        _verify_yolo(tmp_path, "detect")


def test_verify_yolo_rejects_orphan_task_media_and_cache(tmp_path: Path) -> None:
    _write_yolo_dataset(tmp_path, label="0 0.5 0.5 0.2 0.2\n")
    _write_image(tmp_path / "preview.png")
    (tmp_path / "labels.cache").write_text("stale", encoding="utf-8")

    result = _verify_yolo(tmp_path, "detect")

    paths = {item.get("path") for item in result["errors"]}
    assert {"preview.png", "labels.cache"}.issubset(paths)


def test_manifest_rejects_backend_normalization_collisions_before_upload(tmp_path: Path) -> None:
    source = tmp_path / "dataset"
    source.mkdir()
    (source / "bad\\name.txt").write_text("x", encoding="utf-8")

    with pytest.raises(SystemExit, match="canonical NFC POSIX"):
        scan_source_manifest(source, format_name="yolo")


def _write_coco_dataset(root: Path, *, task_key: str, include_task_field: bool = True) -> None:
    _write_image(root / "images" / "sample.png")
    annotation: dict[str, object] = {
        "id": 1,
        "image_id": 1,
        "category_id": 1,
        "bbox": [1, 2, 5, 4],
        "area": 20,
        "iscrowd": 0,
    }
    category: dict[str, object] = {"id": 1, "name": "aircraft"}
    if task_key == "segment" and include_task_field:
        annotation["segmentation"] = [[1, 2, 6, 2, 6, 6, 1, 6]]
    if task_key == "pose":
        category["keypoints"] = ["nose"]
        category["skeleton"] = []
        if include_task_field:
            annotation["keypoints"] = [3, 4, 2]
            annotation["num_keypoints"] = 1
    payload = {
        "images": [{"id": 1, "file_name": "images/sample.png", "width": 16, "height": 12}],
        "categories": [category],
        "annotations": [annotation],
    }
    annotations = root / "annotations"
    annotations.mkdir(parents=True)
    (annotations / "instances.json").write_text(json.dumps(payload), encoding="utf-8")


@pytest.mark.parametrize("task_key", ["detect", "segment", "pose"])
def test_verify_coco_enforces_task_specific_fields(tmp_path: Path, task_key: str) -> None:
    _write_coco_dataset(tmp_path, task_key=task_key)

    result = verify_dataset(source=tmp_path, format_name="coco", task_key=task_key)

    assert result["status"] == "ok"
    assert result["classes"] == ["aircraft"]


@pytest.mark.parametrize("task_key", ["segment", "pose"])
def test_verify_coco_rejects_missing_task_specific_field(tmp_path: Path, task_key: str) -> None:
    _write_coco_dataset(tmp_path, task_key=task_key, include_task_field=False)

    result = verify_dataset(source=tmp_path, format_name="coco", task_key=task_key)

    assert result["status"] == "failed"


def test_verify_coco_json_decode_error_is_structured_with_location(tmp_path: Path) -> None:
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    path = annotations / "broken.json"
    path.write_text('{"images": [\n', encoding="utf-8")

    result = verify_dataset(source=tmp_path, format_name="coco", task_key="detect")

    assert result["errors"] == [
        {
            "code": "invalid_coco_json",
            "path": "annotations/broken.json",
            "line": 2,
            "column": 1,
            "message": "COCO annotation JSON is malformed",
        }
    ]


def test_verify_coco_rejects_control_characters_in_file_name(tmp_path: Path) -> None:
    _write_coco_dataset(tmp_path, task_key="detect")
    annotation_path = tmp_path / "annotations" / "instances.json"
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    payload["images"][0]["file_name"] = "images/bad\nname.png"
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")

    result = verify_dataset(source=tmp_path, format_name="coco", task_key="detect")

    assert any(item["code"] == "invalid_coco_file_name" for item in result["errors"])


def test_verify_coco_resolves_standard_split_directory_file_names(tmp_path: Path) -> None:
    _write_coco_dataset(tmp_path, task_key="detect")
    (tmp_path / "images").rename(tmp_path / "train2017")
    annotation_path = tmp_path / "annotations" / "instances.json"
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    payload["images"][0]["file_name"] = "sample.png"
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")

    result = verify_dataset(source=tmp_path, format_name="coco", task_key="detect")

    assert result["status"] == "ok"


def test_verify_coco_rejects_truncated_uncompressed_rle(tmp_path: Path) -> None:
    _write_coco_dataset(tmp_path, task_key="segment")
    annotation_path = tmp_path / "annotations" / "instances.json"
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    payload["annotations"][0]["segmentation"] = {"size": [12, 16], "counts": [1]}
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")

    result = verify_dataset(source=tmp_path, format_name="coco", task_key="segment")

    assert any(item["code"] == "invalid_coco_segmentation" for item in result["errors"])


def test_verify_coco_accepts_lossless_compressed_rle(tmp_path: Path) -> None:
    _write_coco_dataset(tmp_path, task_key="segment")
    import numpy as np

    mask = np.zeros((12, 16), dtype=np.uint8, order="F")
    mask[2:6, 1:6] = 1
    encoded = mask_utils.encode(mask)
    annotation_path = tmp_path / "annotations" / "instances.json"
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    payload["annotations"][0]["segmentation"] = {
        "size": [12, 16],
        "counts": encoded["counts"].decode("ascii"),
    }
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")

    result = verify_dataset(source=tmp_path, format_name="coco", task_key="segment")

    assert result["status"] == "ok"


@pytest.mark.parametrize("encoding", ["rle", "polygons"])
def test_verify_coco_segment_rejects_disconnected_components(tmp_path: Path, encoding: str) -> None:
    _write_coco_dataset(tmp_path, task_key="segment")
    annotation_path = tmp_path / "annotations" / "instances.json"
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    if encoding == "rle":
        import numpy as np

        mask = np.zeros((12, 16), dtype=np.uint8, order="F")
        mask[1:4, 1:4] = 1
        mask[7:10, 11:14] = 1
        encoded = mask_utils.encode(mask)
        segmentation: object = {
            "size": [12, 16],
            "counts": encoded["counts"].decode("ascii"),
        }
    else:
        segmentation = [
            [1, 1, 1, 3, 3, 3, 3, 1],
            [11, 7, 11, 9, 13, 9, 13, 7],
        ]
    payload["annotations"][0]["segmentation"] = segmentation
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")

    result = verify_dataset(source=tmp_path, format_name="coco", task_key="segment")

    assert result["status"] == "failed"
    assert result["errors"][0]["code"] == "invalid_coco_segmentation"


def test_verify_coco_segment_rejects_mask_with_hole(tmp_path: Path) -> None:
    _write_coco_dataset(tmp_path, task_key="segment")
    import numpy as np

    mask = np.zeros((12, 16), dtype=np.uint8, order="F")
    mask[1:11, 2:14] = 1
    mask[4:8, 6:10] = 0
    encoded = mask_utils.encode(mask)
    annotation_path = tmp_path / "annotations" / "instances.json"
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    payload["annotations"][0]["segmentation"] = {
        "size": [12, 16],
        "counts": encoded["counts"].decode("ascii"),
    }
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")

    result = verify_dataset(source=tmp_path, format_name="coco", task_key="segment")

    assert result["status"] == "failed"
    assert result["errors"][0]["code"] == "invalid_coco_segmentation"


def test_verify_coco_rejects_cross_split_taxonomy_drift(tmp_path: Path) -> None:
    _write_coco_dataset(tmp_path, task_key="detect")
    first = tmp_path / "annotations" / "instances.json"
    second_payload = json.loads(first.read_text(encoding="utf-8"))
    second_payload["categories"][0]["name"] = "plane"
    (tmp_path / "annotations" / "second.json").write_text(
        json.dumps(second_payload), encoding="utf-8"
    )

    result = verify_dataset(source=tmp_path, format_name="coco", task_key="detect")

    assert any(item["code"] == "conflicting_coco_taxonomy" for item in result["errors"])


def test_verify_coco_rejects_image_listed_in_two_splits(tmp_path: Path) -> None:
    _write_coco_dataset(tmp_path, task_key="detect")
    first = tmp_path / "annotations" / "instances.json"
    (tmp_path / "annotations" / "second.json").write_text(
        first.read_text(encoding="utf-8"), encoding="utf-8"
    )

    result = verify_dataset(source=tmp_path, format_name="coco", task_key="detect")

    assert any(item["code"] == "duplicate_coco_image_split" for item in result["errors"])


def test_verify_coco_rejects_self_intersecting_polygon(tmp_path: Path) -> None:
    _write_coco_dataset(tmp_path, task_key="segment")
    annotation_path = tmp_path / "annotations" / "instances.json"
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    payload["annotations"][0]["segmentation"] = [[1, 1, 8, 8, 8, 1, 1, 8]]
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")

    result = verify_dataset(source=tmp_path, format_name="coco", task_key="segment")

    assert any(item["code"] == "invalid_coco_segmentation" for item in result["errors"])


def _write_imagenet_dataset(root: Path) -> None:
    _write_image(root / "train" / "aircraft" / "a.png")
    _write_image(root / "val" / "aircraft" / "b.png")


def test_verify_imagenet_requires_consistent_class_directories(tmp_path: Path) -> None:
    _write_imagenet_dataset(tmp_path)

    result = verify_dataset(source=tmp_path, format_name="imagenet", task_key="classify")

    assert result["status"] == "ok"
    assert result["classes"] == ["aircraft"]


def test_verify_imagenet_rejects_unknown_validation_class(tmp_path: Path) -> None:
    _write_imagenet_dataset(tmp_path)
    _write_image(tmp_path / "val" / "unknown" / "c.png")

    result = verify_dataset(source=tmp_path, format_name="imagenet", task_key="classify")

    assert result["status"] == "failed"


def _write_anomalib_dataset(root: Path) -> None:
    _write_image(root / "train" / "good" / "000.png")
    _write_image(root / "test" / "good" / "000.png")
    _write_image(root / "test" / "broken" / "000.png")
    _write_mask(root / "ground_truth" / "broken" / "000_mask.png")


def test_verify_anomalib_accepts_complete_mvtec_structure(tmp_path: Path) -> None:
    _write_anomalib_dataset(tmp_path)

    result = verify_dataset(source=tmp_path, format_name="anomalib", task_key="ad")

    assert result["status"] == "ok"
    assert result["warning_count"] == 0


def test_verify_anomalib_accepts_explicit_provenance_documents(tmp_path: Path) -> None:
    _write_anomalib_dataset(tmp_path)
    (tmp_path / "license.txt").write_text("MVTec license\n", encoding="utf-8")
    (tmp_path / "source_records.json").write_text("{}\n", encoding="utf-8")

    result = verify_dataset(source=tmp_path, format_name="anomalib", task_key="ad")

    assert result["status"] == "ok"


def test_verify_anomalib_accepts_generic_bad_mask_with_same_stem(tmp_path: Path) -> None:
    _write_image(tmp_path / "train" / "good" / "000.png")
    _write_image(tmp_path / "test" / "good" / "000.png")
    _write_image(tmp_path / "test" / "bad" / "000.png")
    _write_mask(tmp_path / "ground_truth" / "bad" / "000.png")

    result = verify_dataset(source=tmp_path, format_name="anomalib", task_key="ad")

    assert result["status"] == "ok"


def test_verify_anomalib_rejects_missing_bad_mask(tmp_path: Path) -> None:
    _write_anomalib_dataset(tmp_path)
    (tmp_path / "ground_truth" / "broken" / "000_mask.png").unlink()

    result = verify_dataset(source=tmp_path, format_name="anomalib", task_key="ad")

    assert result["status"] == "failed"
    assert result["errors"][0]["code"] == "missing_anomaly_mask"


def test_verify_anomalib_rejects_mismatched_mask_dimensions(tmp_path: Path) -> None:
    _write_anomalib_dataset(tmp_path)
    _write_mask(root := tmp_path / "ground_truth" / "broken" / "000_mask.png", size=(8, 8))
    assert root.exists()

    result = verify_dataset(source=tmp_path, format_name="anomalib", task_key="ad")

    assert result["status"] == "failed"
    assert result["errors"][0]["code"] == "anomaly_mask_size_mismatch"


def test_verify_anomalib_rejects_orphan_mask(tmp_path: Path) -> None:
    _write_anomalib_dataset(tmp_path)
    _write_mask(tmp_path / "ground_truth" / "broken" / "orphan_mask.png")

    result = verify_dataset(source=tmp_path, format_name="anomalib", task_key="ad")

    assert result["status"] == "failed"
    assert any(error["code"] == "orphan_anomaly_mask" for error in result["errors"])


def test_verify_anomalib_rejects_images_outside_exact_split_roles(tmp_path: Path) -> None:
    _write_anomalib_dataset(tmp_path)
    _write_image(tmp_path / "train" / "broken" / "unexpected.png")
    _write_image(tmp_path / "validation" / "broken" / "unexpected.png")

    result = verify_dataset(source=tmp_path, format_name="anomalib", task_key="ad")

    paths = {
        item.get("path")
        for item in result["errors"]
        if item["code"] == "unexpected_anomalib_member"
    }
    assert paths == {
        "train/broken/unexpected.png",
        "validation/broken/unexpected.png",
    }


def test_verify_anomalib_rejects_malformed_image(tmp_path: Path) -> None:
    _write_anomalib_dataset(tmp_path)
    (tmp_path / "train" / "good" / "000.png").write_bytes(b"")

    result = verify_dataset(source=tmp_path, format_name="anomalib", task_key="ad")

    assert result["status"] == "failed"
    assert any(error["code"] == "invalid_image" for error in result["errors"])


def test_verify_anomalib_rejects_rgb_mask(tmp_path: Path) -> None:
    _write_anomalib_dataset(tmp_path)
    Image.new("RGB", (16, 12), color=(255, 0, 0)).save(
        tmp_path / "ground_truth" / "broken" / "000_mask.png"
    )

    result = verify_dataset(source=tmp_path, format_name="anomalib", task_key="ad")

    assert any(error["code"] == "invalid_anomaly_mask" for error in result["errors"])
