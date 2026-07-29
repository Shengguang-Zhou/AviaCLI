from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
import yaml
from PIL import Image
from pycocotools import mask as mask_utils

from avia_cli.core.uploads.inspect import inspect_dataset, verify_dataset
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


def test_verify_yolo_accepts_tiff_images(tmp_path: Path) -> None:
    _write_image(tmp_path / "images" / "train" / "sample.tiff")
    label_path = tmp_path / "labels" / "train" / "sample.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("0 0.5 0.5 0.25 0.25\n", encoding="utf-8")
    (tmp_path / "data.yaml").write_text("names: [aircraft]\n", encoding="utf-8")

    result = _verify_yolo(tmp_path, "detect")

    assert result["status"] == "ok"
    manifest = scan_source_manifest(tmp_path, format_name="yolo")
    tiff_item = next(item for item in manifest["files"] if item["relative_path"].endswith(".tiff"))
    assert (tiff_item["width"], tiff_item["height"]) == (16, 12)
    assert "content_type" not in tiff_item


def test_verify_yolo_preserves_supported_uppercase_image_suffix(tmp_path: Path) -> None:
    _write_yolo_dataset(tmp_path, label="0 0.5 0.5 0.25 0.25\n")
    (tmp_path / "images" / "train" / "sample.png").rename(
        tmp_path / "images" / "train" / "sample.PNG"
    )

    result = _verify_yolo(tmp_path, "detect")

    assert result["status"] == "ok"
    manifest = scan_source_manifest(tmp_path, format_name="yolo")
    assert "images/train/sample.PNG" in {item["relative_path"] for item in manifest["files"]}


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


def test_verify_coco_preserves_supported_uppercase_image_suffix(tmp_path: Path) -> None:
    _write_coco_dataset(tmp_path, task_key="detect")
    (tmp_path / "images" / "sample.png").rename(tmp_path / "images" / "sample.PNG")
    annotation_path = tmp_path / "annotations" / "instances.json"
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    payload["images"][0]["file_name"] = "images/sample.PNG"
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")

    result = verify_dataset(source=tmp_path, format_name="coco", task_key="detect")

    assert result["status"] == "ok"


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


def test_verify_coco_rejects_basename_lookup_across_split_directories(tmp_path: Path) -> None:
    _write_coco_dataset(tmp_path, task_key="detect")
    (tmp_path / "images").rename(tmp_path / "train2017")
    annotation_path = tmp_path / "annotations" / "instances.json"
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    payload["images"][0]["file_name"] = "sample.png"
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")

    result = verify_dataset(source=tmp_path, format_name="coco", task_key="detect")

    assert result["status"] == "failed"
    assert any(
        item["code"] == "invalid_image" and item["file_name"] == "sample.png"
        for item in result["errors"]
    )
    assert any(
        item["code"] == "orphan_coco_image" and item["path"] == "train2017/sample.png"
        for item in result["errors"]
    )


def test_verify_coco_does_not_resolve_images_from_hidden_client_state(tmp_path: Path) -> None:
    _write_coco_dataset(tmp_path, task_key="detect")
    hidden_image = tmp_path / "images" / ".avia" / "sample.png"
    hidden_image.parent.mkdir(parents=True)
    (tmp_path / "images" / "sample.png").replace(hidden_image)
    annotation_path = tmp_path / "annotations" / "instances.json"
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    payload["images"][0]["file_name"] = "images/.avia/sample.png"
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")

    result = verify_dataset(source=tmp_path, format_name="coco", task_key="detect")

    assert result["status"] == "failed"
    assert any(item["code"] == "invalid_image" for item in result["errors"])


@pytest.mark.parametrize(
    ("task_key", "field", "value", "error_code"),
    [
        ("detect", "bbox", [True, "2", 5, 4], "invalid_coco_bbox"),
        (
            "segment",
            "segmentation",
            [[1, "2", 6, 2, 6, 6, 1, 6]],
            "invalid_coco_segmentation",
        ),
        ("pose", "keypoints", [3, "4", 2], "invalid_coco_keypoints"),
    ],
)
def test_verify_coco_requires_json_numbers_without_bool_or_string_coercion(
    tmp_path: Path,
    task_key: str,
    field: str,
    value: object,
    error_code: str,
) -> None:
    _write_coco_dataset(tmp_path, task_key=task_key)
    annotation_path = tmp_path / "annotations" / "instances.json"
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    payload["annotations"][0][field] = value
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")

    result = verify_dataset(source=tmp_path, format_name="coco", task_key=task_key)

    assert result["status"] == "failed"
    assert any(item["code"] == error_code for item in result["errors"])


@pytest.mark.parametrize(
    ("task_key", "field", "value", "error_code"),
    [
        ("detect", "bbox", [10**400, 2, 5, 4], "invalid_coco_bbox"),
        (
            "segment",
            "segmentation",
            [[1, 2, 10**400, 2, 6, 6, 1, 6]],
            "invalid_coco_segmentation",
        ),
        ("pose", "keypoints", [10**400, 4, 2], "invalid_coco_keypoints"),
    ],
)
def test_verify_coco_rejects_integers_that_overflow_float_without_traceback(
    tmp_path: Path,
    task_key: str,
    field: str,
    value: object,
    error_code: str,
) -> None:
    _write_coco_dataset(tmp_path, task_key=task_key)
    annotation_path = tmp_path / "annotations" / "instances.json"
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    payload["annotations"][0][field] = value
    annotation_path.write_text(json.dumps(payload), encoding="utf-8")

    result = verify_dataset(source=tmp_path, format_name="coco", task_key=task_key)

    assert result["status"] == "failed"
    assert any(item["code"] == error_code for item in result["errors"])


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


def test_verify_imagenet_preserves_supported_uppercase_image_suffix(tmp_path: Path) -> None:
    _write_imagenet_dataset(tmp_path)
    (tmp_path / "train" / "aircraft" / "a.png").rename(tmp_path / "train" / "aircraft" / "a.PNG")
    (tmp_path / "val" / "aircraft" / "b.png").rename(tmp_path / "val" / "aircraft" / "b.PNG")

    result = verify_dataset(source=tmp_path, format_name="imagenet", task_key="classify")

    assert result["status"] == "ok"


def test_verify_imagenet_indexes_images_once_instead_of_scanning_per_class(
    tmp_path: Path,
) -> None:
    from avia_cli.core.uploads.inventory import DatasetRoleInventory
    from avia_cli.core.uploads.validation_folders import validate_imagenet

    class CountingImagePaths:
        def __init__(self, values: tuple[str, ...]) -> None:
            self.values = values
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            return iter(self.values)

    class_count = 12
    image_paths: list[str] = []
    for index in range(class_count):
        class_name = f"class-{index:02d}"
        for split in ("train", "val"):
            relative = f"{split}/{class_name}/{split}.png"
            image_paths.append(relative)
            _write_image(tmp_path / relative)
    counted_paths = CountingImagePaths(tuple(image_paths))
    inventory = DatasetRoleInventory(
        format_name="imagenet",
        image_paths=counted_paths,  # type: ignore[arg-type]
        label_paths=(),
        mask_paths=(),
    )

    classes, errors = validate_imagenet(tmp_path, inventory=inventory)

    assert len(classes) == class_count
    assert errors == []
    assert counted_paths.iterations == 2


def test_verify_imagenet_does_not_count_images_from_hidden_client_state(tmp_path: Path) -> None:
    _write_image(tmp_path / "train" / "aircraft" / ".avia" / "only.png")
    _write_image(tmp_path / "val" / "aircraft" / ".avia" / "only.png")

    result = verify_dataset(source=tmp_path, format_name="imagenet", task_key="classify")

    assert result["status"] == "failed"
    assert [item["code"] for item in result["errors"]].count("empty_imagenet_class") == 2


def test_verify_imagenet_ignores_hidden_client_state_role_directories(tmp_path: Path) -> None:
    _write_imagenet_dataset(tmp_path)
    _write_image(tmp_path / "train" / ".avia" / "hidden.png")
    _write_image(tmp_path / "val" / ".avia" / "hidden.png")

    result = verify_dataset(source=tmp_path, format_name="imagenet", task_key="classify")

    assert result["status"] == "ok"
    assert result["classes"] == ["aircraft"]


def test_verify_imagenet_rejects_unknown_validation_class(tmp_path: Path) -> None:
    _write_imagenet_dataset(tmp_path)
    _write_image(tmp_path / "val" / "unknown" / "c.png")

    result = verify_dataset(source=tmp_path, format_name="imagenet", task_key="classify")

    assert result["status"] == "failed"


def _write_anomalib_dataset(root: Path) -> None:
    _write_image(root / "train" / "good" / "train.png")
    _write_image(root / "val" / "good" / "val-good.png")
    _write_image(root / "val" / "bad" / "val-bad.jpg")
    _write_image(root / "test" / "good" / "test-good.png")
    _write_image(root / "test" / "bad" / "test-bad.webp")
    _write_mask(root / "ground_truth" / "val" / "bad" / "val-bad.png")
    _write_mask(root / "ground_truth" / "test" / "bad" / "test-bad.png")


def _write_mvtec_small_corpus(root: Path) -> None:
    role_counts = {
        ("train", "good"): 8,
        ("val", "good"): 2,
        ("val", "bad"): 3,
        ("test", "good"): 4,
        ("test", "bad"): 3,
    }
    for (split, class_name), count in role_counts.items():
        for index in range(count):
            stem = f"{split}-{class_name}-{index:02d}"
            _write_image(root / split / class_name / f"{stem}.png")
            if class_name == "bad":
                _write_mask(root / "ground_truth" / split / "bad" / f"{stem}.png")


def test_verify_anomalib_accepts_only_complete_training_structure(tmp_path: Path) -> None:
    _write_anomalib_dataset(tmp_path)

    result = verify_dataset(source=tmp_path, format_name="anomalib", task_key="ad")

    assert result["status"] == "ok"
    assert result["classes"] == ["good", "bad"]
    assert result["image_count"] == 5
    assert result["label_count"] == 0
    assert result["mask_count"] == 2
    assert result["warning_count"] == 0


def test_mvtec_small_corpus_has_one_canonical_role_inventory(tmp_path: Path) -> None:
    _write_mvtec_small_corpus(tmp_path)

    manifest = scan_source_manifest(tmp_path, format_name="anomalib")
    inspected = inspect_dataset(source=tmp_path, format_name="anomalib", task_key="ad")
    verified = verify_dataset(source=tmp_path, format_name="anomalib", task_key="ad")

    assert verified["status"] == "ok"
    for payload in (manifest, inspected, verified):
        assert payload["image_count"] == 20
        assert payload["label_count"] == 0
        assert payload["mask_count"] == 6


def test_non_ad_formats_keep_format_aware_inventory_counts(tmp_path: Path) -> None:
    yolo = tmp_path / "yolo"
    coco = tmp_path / "coco"
    imagenet = tmp_path / "imagenet"
    _write_yolo_dataset(yolo, label="0 0.5 0.5 0.25 0.25\n")
    _write_coco_dataset(coco, task_key="detect")
    _write_imagenet_dataset(imagenet)

    expected = {
        "yolo": (yolo, "detect", 1, 1),
        "coco": (coco, "detect", 1, 1),
        "imagenet": (imagenet, "classify", 2, 0),
    }
    for format_name, (source, task_key, image_count, label_count) in expected.items():
        result = verify_dataset(
            source=source,
            format_name=format_name,
            task_key=task_key,
        )
        assert result["status"] == "ok"
        assert result["image_count"] == image_count
        assert result["label_count"] == label_count
        assert result["mask_count"] == 0


@pytest.mark.parametrize("suffix", (".bmp", ".tif", ".tiff"))
def test_verify_anomalib_rejects_undocumented_source_image_suffixes(
    tmp_path: Path,
    suffix: str,
) -> None:
    _write_anomalib_dataset(tmp_path)
    (tmp_path / "val" / "bad" / "val-bad.jpg").unlink()
    relative_path = f"val/bad/val-bad{suffix}"
    _write_image(tmp_path / relative_path)

    result = verify_dataset(source=tmp_path, format_name="anomalib", task_key="ad")

    assert result["status"] == "failed"
    assert any(
        item["code"] == "unexpected_anomalib_member" and item.get("path") == relative_path
        for item in result["errors"]
    )


@pytest.mark.parametrize(
    ("relative_path", "mode", "color", "expected_format"),
    [
        ("val/good/val-good.png", "RGB", "white", "PNG"),
        ("val/bad/val-bad.jpg", "RGB", "white", "JPEG"),
        ("test/bad/test-bad.webp", "RGB", "white", "WEBP"),
        ("ground_truth/val/bad/val-bad.png", "L", 255, "PNG"),
    ],
)
def test_verify_anomalib_rejects_tiff_bytes_disguised_as_canonical_members(
    tmp_path: Path,
    relative_path: str,
    mode: str,
    color: str | int,
    expected_format: str,
) -> None:
    _write_anomalib_dataset(tmp_path)
    Image.new(mode, (16, 12), color=color).save(tmp_path / relative_path, format="TIFF")

    result = verify_dataset(source=tmp_path, format_name="anomalib", task_key="ad")

    assert result["status"] == "failed"
    assert result["errors"] == [
        {
            "code": "invalid_image",
            "message": (
                "image encoding does not match its declared suffix: "
                f"path={relative_path} expected={expected_format} actual=TIFF"
            ),
            "path": relative_path,
            "expected_format": expected_format,
            "actual_format": "TIFF",
        }
    ]


def test_inspect_anomalib_reports_canonical_binary_taxonomy(tmp_path: Path) -> None:
    _write_anomalib_dataset(tmp_path)

    result = inspect_dataset(source=tmp_path, format_name="anomalib", task_key="ad")

    assert result["classes"] == ["good", "bad"]
    assert result["image_count"] == 5
    assert result["label_count"] == 0
    assert result["mask_count"] == 2


@pytest.mark.parametrize("inspect", [inspect_dataset, verify_dataset])
def test_anomalib_scan_failure_preserves_canonical_binary_taxonomy(
    tmp_path: Path, inspect: Callable[..., dict[str, object]]
) -> None:
    _write_anomalib_dataset(tmp_path)
    (tmp_path / "train" / "good" / "000.png").write_bytes(b"not-an-image")

    result = inspect(source=tmp_path, format_name="anomalib", task_key="ad")

    assert result["status"] == "failed"
    assert result["classes"] == ["good", "bad"]
    assert result["errors"][0]["code"] == "invalid_image"


def test_verify_anomalib_ignores_hidden_client_state_directory(tmp_path: Path) -> None:
    _write_anomalib_dataset(tmp_path)
    _write_image(tmp_path / "test" / ".avia" / "hidden.png")

    result = verify_dataset(source=tmp_path, format_name="anomalib", task_key="ad")

    assert result["status"] == "ok"
    assert result["classes"] == ["good", "bad"]


@pytest.mark.parametrize("filename", ["README", "LICENSE", "source_records.json"])
def test_verify_anomalib_accepts_exact_provenance_document_names(
    tmp_path: Path,
    filename: str,
) -> None:
    _write_anomalib_dataset(tmp_path)
    (tmp_path / filename).write_text("{}\n", encoding="utf-8")

    result = verify_dataset(source=tmp_path, format_name="anomalib", task_key="ad")

    assert result["status"] == "ok"


@pytest.mark.parametrize(
    "filename",
    [
        "README.txt",
        "README.md",
        "readme",
        "license.txt",
        "LICENSE.txt",
        "SOURCE_RECORDS.JSON",
        "notes.txt",
        "provenance.json",
    ],
)
def test_verify_anomalib_rejects_noncanonical_provenance_document_names(
    tmp_path: Path,
    filename: str,
) -> None:
    _write_anomalib_dataset(tmp_path)
    (tmp_path / filename).write_text("{}\n", encoding="utf-8")

    result = verify_dataset(source=tmp_path, format_name="anomalib", task_key="ad")

    assert result["status"] == "failed"
    assert any(
        item["code"] == "unexpected_anomalib_member" and item.get("path") == filename
        for item in result["errors"]
    )


def test_verify_anomalib_rejects_missing_bad_mask(tmp_path: Path) -> None:
    _write_anomalib_dataset(tmp_path)
    (tmp_path / "ground_truth" / "val" / "bad" / "val-bad.png").unlink()

    result = verify_dataset(source=tmp_path, format_name="anomalib", task_key="ad")

    assert result["status"] == "failed"
    assert result["errors"][0]["code"] == "missing_anomaly_mask"


def test_verify_anomalib_rejects_mismatched_mask_dimensions(tmp_path: Path) -> None:
    _write_anomalib_dataset(tmp_path)
    _write_mask(
        root := tmp_path / "ground_truth" / "test" / "bad" / "test-bad.png",
        size=(8, 8),
    )
    assert root.exists()

    result = verify_dataset(source=tmp_path, format_name="anomalib", task_key="ad")

    assert result["status"] == "failed"
    assert result["errors"][0]["code"] == "anomaly_mask_size_mismatch"


def test_verify_anomalib_rejects_orphan_mask(tmp_path: Path) -> None:
    _write_anomalib_dataset(tmp_path)
    _write_mask(tmp_path / "ground_truth" / "test" / "bad" / "orphan.png")

    result = verify_dataset(source=tmp_path, format_name="anomalib", task_key="ad")

    assert result["status"] == "failed"
    assert any(error["code"] == "orphan_anomaly_mask" for error in result["errors"])


@pytest.mark.parametrize(
    "relative_path",
    [
        "train/bad/unexpected.png",
        "validation/good/unexpected.png",
        "validation/bad/unexpected.png",
        "test/crack/unexpected.png",
        "ground_truth/crack/unexpected_mask.png",
        "ground_truth/test/bad/test-bad_mask.png",
        "test/good/nested/unexpected.png",
        "test/bad/unsupported.gif",
    ],
)
def test_verify_anomalib_rejects_retired_or_noncanonical_layout_members(
    tmp_path: Path,
    relative_path: str,
) -> None:
    _write_anomalib_dataset(tmp_path)
    _write_image(tmp_path / relative_path)

    result = verify_dataset(source=tmp_path, format_name="anomalib", task_key="ad")

    assert result["status"] == "failed"
    assert any(
        item["code"] == "unexpected_anomalib_member" and item.get("path") == relative_path
        for item in result["errors"]
    )


def test_verify_anomalib_rejects_uppercase_image_suffix_before_validation(
    tmp_path: Path,
) -> None:
    _write_anomalib_dataset(tmp_path)
    path = "ground_truth/test/bad/test-bad.PNG"
    _write_image(tmp_path / path)

    with pytest.raises(SystemExit, match='"code": "invalid_dataset_path"'):
        verify_dataset(source=tmp_path, format_name="anomalib", task_key="ad")


def test_verify_anomalib_rejects_uppercase_source_suffix_before_validation(
    tmp_path: Path,
) -> None:
    _write_anomalib_dataset(tmp_path)
    (tmp_path / "val" / "bad" / "val-bad.jpg").rename(tmp_path / "val" / "bad" / "val-bad.JPG")

    with pytest.raises(SystemExit, match='"code": "invalid_dataset_path"'):
        verify_dataset(source=tmp_path, format_name="anomalib", task_key="ad")


def test_verify_anomalib_rejects_unexpected_empty_directory(tmp_path: Path) -> None:
    _write_anomalib_dataset(tmp_path)
    (tmp_path / "validation" / "good").mkdir(parents=True)

    result = verify_dataset(source=tmp_path, format_name="anomalib", task_key="ad")

    assert result["status"] == "failed"
    assert any(
        item["code"] == "unexpected_anomalib_directory" and item.get("path") == "validation/good"
        for item in result["errors"]
    )


def test_verify_anomalib_reports_missing_canonical_directory(tmp_path: Path) -> None:
    _write_anomalib_dataset(tmp_path)
    (tmp_path / "val" / "good" / "val-good.png").unlink()
    (tmp_path / "val" / "good").rmdir()

    result = verify_dataset(source=tmp_path, format_name="anomalib", task_key="ad")

    assert result["status"] == "failed"
    assert any(
        item["code"] == "missing_anomalib_directory" and item.get("path") == "val/good"
        for item in result["errors"]
    )


@pytest.mark.parametrize(
    ("relative_path", "expected_split", "expected_class"),
    [
        ("val/good/val-good.png", "val", "good"),
        ("val/bad/val-bad.jpg", "val", "bad"),
        ("test/good/test-good.png", "test", "good"),
        ("test/bad/test-bad.webp", "test", "bad"),
    ],
)
def test_verify_anomalib_rejects_empty_required_evaluation_role(
    tmp_path: Path,
    relative_path: str,
    expected_split: str,
    expected_class: str,
) -> None:
    _write_anomalib_dataset(tmp_path)
    (tmp_path / relative_path).unlink()

    result = verify_dataset(source=tmp_path, format_name="anomalib", task_key="ad")

    assert result["status"] == "failed"
    assert any(
        item["code"] == "empty_anomalib_role"
        and item.get("split") == expected_split
        and item.get("class_name") == expected_class
        for item in result["errors"]
    )


def test_verify_anomalib_rejects_duplicate_source_stems(tmp_path: Path) -> None:
    _write_anomalib_dataset(tmp_path)
    _write_image(tmp_path / "val" / "bad" / "val-bad.webp")

    result = verify_dataset(source=tmp_path, format_name="anomalib", task_key="ad")

    assert result["status"] == "failed"
    assert any(
        item["code"] == "duplicate_anomalib_source_stem"
        and item.get("split") == "val"
        and item.get("class_name") == "bad"
        and item.get("stem") == "val-bad"
        for item in result["errors"]
    )


def test_verify_anomalib_rejects_malformed_image(tmp_path: Path) -> None:
    _write_anomalib_dataset(tmp_path)
    (tmp_path / "train" / "good" / "000.png").write_bytes(b"")

    result = verify_dataset(source=tmp_path, format_name="anomalib", task_key="ad")

    assert result["status"] == "failed"
    assert any(error["code"] == "invalid_image" for error in result["errors"])


def test_verify_anomalib_rejects_rgb_mask(tmp_path: Path) -> None:
    _write_anomalib_dataset(tmp_path)
    Image.new("RGB", (16, 12), color=(255, 0, 0)).save(
        tmp_path / "ground_truth" / "val" / "bad" / "val-bad.png"
    )

    result = verify_dataset(source=tmp_path, format_name="anomalib", task_key="ad")

    assert any(error["code"] == "invalid_anomaly_mask" for error in result["errors"])
