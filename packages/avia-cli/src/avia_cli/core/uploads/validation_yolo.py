from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from avia_cli.core.uploads.class_catalog import require_canonical_class_catalog
from avia_cli.core.uploads.inventory import DatasetRoleInventory
from avia_cli.core.uploads.metadata import read_yolo_metadata
from avia_cli.core.uploads.validation_common import (
    error,
    finite_numbers,
    image_size,
    is_cache_path,
    is_document_path,
    is_weakly_simple_polygon,
    normalized_points,
    polygon_area,
)

_YOLO_METADATA_NAMES = {"data.yaml", "data.yml", "dataset.yaml", "dataset.yml", "classes.txt"}
_NORMALIZED_ROUNDING_TOLERANCE = 1e-6


def validate_yolo(
    *,
    source_root: Path,
    manifest: dict[str, object],
    inventory: DatasetRoleInventory,
    task_key: str,
    declared_classes: list[str] | None = None,
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    try:
        metadata = read_yolo_metadata(source_root)
    except SystemExit as exc:
        metadata = {}
        errors.append(error("invalid_yolo_metadata", str(exc)))
    metadata_classes = list(metadata.get("names") or [])
    classes = _resolve_classes(
        metadata_classes=metadata_classes,
        declared_classes=declared_classes,
        errors=errors,
    )
    if not classes:
        errors.append(error("missing_class_names", "YOLO class names are required"))

    files = [dict(item) for item in list(manifest.get("files") or []) if isinstance(item, dict)]
    file_by_relative = {str(item.get("relative_path") or ""): item for item in files}
    relative_paths = set(file_by_relative)
    image_paths = list(inventory.image_paths)
    label_paths = list(inventory.label_paths)
    role_paths = set(image_paths) | set(label_paths)
    for relative_path in sorted(relative_paths):
        allowed = (
            relative_path in role_paths
            or relative_path in _YOLO_METADATA_NAMES
            or is_document_path(relative_path)
        )
        if not allowed or is_cache_path(relative_path):
            errors.append(
                error(
                    "unexpected_yolo_member",
                    "YOLO datasets may contain only task images, labels, one metadata file, and explicit documentation",
                    path=relative_path,
                )
            )
    if not image_paths:
        errors.append(error("no_images", "YOLO dataset has no images"))
    if task_key == "pose":
        _validate_pose_metadata(metadata.get("kpt_shape"), errors=errors)
    expected_labels = {_label_path_for_image(path) for path in image_paths}
    label_dimensions: dict[str, tuple[int, int]] = {}
    for image_path in image_paths:
        try:
            actual_width, actual_height = image_size(source_root / image_path)
        except (OSError, ValueError) as exc:
            errors.append(error("invalid_image", str(exc), path=image_path))
        else:
            label_dimensions[_label_path_for_image(image_path)] = (
                actual_width,
                actual_height,
            )
            manifest_item = file_by_relative[image_path]
            declared_width = int(manifest_item.get("width") or 0)
            declared_height = int(manifest_item.get("height") or 0)
            if (declared_width and declared_width != actual_width) or (
                declared_height and declared_height != actual_height
            ):
                errors.append(
                    error(
                        "yolo_image_size_mismatch",
                        "manifest image dimensions do not match the decoded file",
                        path=image_path,
                        manifest_size=[declared_width, declared_height],
                        decoded_size=[actual_width, actual_height],
                    )
                )
        expected = _label_path_for_image(image_path)
        if expected not in relative_paths:
            errors.append(
                error(
                    "missing_yolo_label",
                    "image has no matching YOLO label file",
                    path=image_path,
                    expected_label_path=expected,
                )
            )
    for label_path in label_paths:
        if label_path not in expected_labels:
            errors.append(
                error(
                    "orphan_yolo_label",
                    "label has no matching image file",
                    path=label_path,
                )
            )
        _validate_label_file(
            path=source_root / label_path,
            relative_path=label_path,
            task_key=task_key,
            class_count=len(classes),
            kpt_shape=metadata.get("kpt_shape"),
            image_dimensions=label_dimensions.get(label_path),
            errors=errors,
            warnings=warnings,
        )
    return classes, errors, warnings


def _resolve_classes(
    *,
    metadata_classes: list[str],
    declared_classes: list[str] | None,
    errors: list[dict[str, Any]],
) -> list[str]:
    if declared_classes is None:
        return metadata_classes
    try:
        canonical_declared_classes = require_canonical_class_catalog(
            declared_classes,
            label="--class values",
        )
    except ValueError as exc:
        errors.append(
            error(
                "invalid_declared_class_names",
                str(exc),
            )
        )
        return []
    if metadata_classes and canonical_declared_classes != metadata_classes:
        errors.append(
            error(
                "conflicting_class_names",
                "--class values must exactly match YOLO dataset metadata when both are present",
                metadata_classes=metadata_classes,
                declared_classes=canonical_declared_classes,
            )
        )
    return canonical_declared_classes


def _label_path_for_image(image_path: str) -> str:
    stem = Path(image_path).with_suffix("").as_posix()
    return f"labels/{stem[len('images/') :]}.txt"


def _validate_label_file(
    *,
    path: Path,
    relative_path: str,
    task_key: str,
    class_count: int,
    kpt_shape: object,
    image_dimensions: tuple[int, int] | None,
    errors: list[dict[str, Any]],
    warnings: list[dict[str, Any]],
) -> None:
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        errors.append(error("label_read_failed", str(exc), path=relative_path))
        return
    lines = [
        (line_number, line.strip()) for line_number, line in enumerate(raw_lines, 1) if line.strip()
    ]
    seen_classes: set[int] = set()
    for line_number, line in lines:
        parts = line.split()
        class_id = _class_id(
            parts=parts,
            class_count=class_count,
            path=relative_path,
            line_number=line_number,
            errors=errors,
        )
        if class_id is None:
            continue
        if task_key == "classify":
            if len(parts) != 1:
                errors.append(
                    _row_error(task_key, relative_path, line_number, "row must be one class id")
                )
            elif class_id in seen_classes:
                errors.append(
                    error(
                        "duplicate_yolo_class",
                        "multilabel classification files are sets and cannot repeat class ids",
                        path=relative_path,
                        line=line_number,
                        class_id=class_id,
                    )
                )
            seen_classes.add(class_id)
            continue
        values = finite_numbers(parts[1:])
        if values is None:
            errors.append(
                _row_error(
                    task_key, relative_path, line_number, "coordinates must be finite numbers"
                )
            )
            continue
        message = _validate_values(
            task_key=task_key,
            values=values,
            kpt_shape=kpt_shape,
            image_dimensions=image_dimensions,
        )
        if message:
            errors.append(_row_error(task_key, relative_path, line_number, message))
        elif task_key == "segment":
            points = normalized_points(values)
            if points is not None and not is_weakly_simple_polygon(points):
                warnings.append(
                    error(
                        "yolo_segment_topology",
                        "segment is rasterizable but contains crossing or non-canonical bridge topology",
                        path=relative_path,
                        line=line_number,
                    )
                )


def _class_id(
    *,
    parts: list[str],
    class_count: int,
    path: str,
    line_number: int,
    errors: list[dict[str, Any]],
) -> int | None:
    if not parts or not parts[0].isascii() or not parts[0].isdigit():
        errors.append(
            error(
                "invalid_yolo_class",
                "YOLO class id must be a non-negative integer",
                path=path,
                line=line_number,
            )
        )
        return None
    class_id = int(parts[0])
    if class_id >= class_count:
        errors.append(
            error(
                "unknown_yolo_class",
                "YOLO class id is outside declared class names",
                path=path,
                line=line_number,
                class_id=class_id,
            )
        )
        return None
    return class_id


def _validate_values(
    *,
    task_key: str,
    values: list[float],
    kpt_shape: object,
    image_dimensions: tuple[int, int] | None,
) -> str | None:
    if task_key == "detect":
        return _validate_box(values)
    if task_key == "segment":
        if len(values) < 6 or len(values) % 2:
            return "row must contain at least three x/y polygon points"
        points = normalized_points(values)
        if points is None:
            return "polygon coordinates must be normalized to [0, 1]"
        return _validate_segment_geometry(points, image_dimensions=image_dimensions)
    if task_key == "obb":
        if len(values) != 8:
            return "row must contain exactly four x/y points"
        points = normalized_points(values)
        if points is None:
            return "quadrilateral coordinates must be normalized to [0, 1]"
        if not _strictly_convex(points):
            return "quadrilateral must be convex, non-self-intersecting, and non-zero"
        return None
    if task_key == "pose":
        return _validate_pose(values, kpt_shape)
    return f"unsupported YOLO task: {task_key}"


def _validate_segment_geometry(
    points: list[tuple[float, float]],
    *,
    image_dimensions: tuple[int, int] | None,
) -> str | None:
    coordinates = np.asarray(points, dtype=np.float32)
    hull = cv2.convexHull(coordinates)
    if len(hull) < 3 or float(cv2.contourArea(hull)) <= 1e-12:
        return "polygon area must be non-zero"
    if image_dimensions is None:
        return None
    width, height = image_dimensions
    scale = np.asarray([max(width - 1, 0), max(height - 1, 0)], dtype=np.float32)
    pixels = np.rint(coordinates * scale).astype(np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [pixels], 1)
    if not np.any(mask):
        return "polygon must rasterize to a non-empty mask"
    return None


def _validate_box(values: list[float]) -> str | None:
    if len(values) != 4:
        return "row must contain exactly cx cy width height"
    if any(value < 0.0 or value > 1.0 for value in values):
        return "box coordinates must be normalized to [0, 1]"
    if values[2] <= 0.0 or values[3] <= 0.0:
        return "box width and height must be positive"
    center_x, center_y, width, height = values
    if (
        center_x - width / 2 < -_NORMALIZED_ROUNDING_TOLERANCE
        or center_x + width / 2 > 1 + _NORMALIZED_ROUNDING_TOLERANCE
        or center_y - height / 2 < -_NORMALIZED_ROUNDING_TOLERANCE
        or center_y + height / 2 > 1 + _NORMALIZED_ROUNDING_TOLERANCE
    ):
        return "box corners must fit inside normalized image bounds"
    return None


def _validate_pose(values: list[float], kpt_shape: object) -> str | None:
    if (
        not isinstance(kpt_shape, list)
        or len(kpt_shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in kpt_shape)
        or int(kpt_shape[0]) <= 0
        or int(kpt_shape[0]) > 2048
        or int(kpt_shape[1]) != 3
    ):
        return "pose metadata must declare exact kpt_shape=[K,3] with 1 <= K <= 2048"
    keypoint_count, dimensions = int(kpt_shape[0]), int(kpt_shape[1])
    expected = 4 + keypoint_count * dimensions
    if len(values) != expected:
        return f"pose row must contain box plus {keypoint_count}x{dimensions} keypoint values"
    box_error = _validate_box(values[:4])
    if box_error:
        return box_error
    keypoints = values[4:]
    for offset in range(0, len(keypoints), dimensions):
        x, y = keypoints[offset : offset + 2]
        if x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0:
            return "keypoint x/y coordinates must be normalized to [0, 1]"
        if keypoints[offset + 2] not in {0.0, 1.0, 2.0}:
            return "keypoint visibility must be exactly 0, 1, or 2"
    return None


def _validate_pose_metadata(kpt_shape: object, *, errors: list[dict[str, Any]]) -> None:
    if (
        not isinstance(kpt_shape, list)
        or len(kpt_shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in kpt_shape)
        or int(kpt_shape[0]) <= 0
        or int(kpt_shape[0]) > 2048
        or int(kpt_shape[1]) != 3
    ):
        errors.append(
            error(
                "invalid_yolo_pose_metadata",
                "pose datasets require exact kpt_shape=[K,3] metadata with 1 <= K <= 2048",
            )
        )


def _strictly_convex(points: list[tuple[float, float]]) -> bool:
    if len(points) != 4 or polygon_area(points) <= 1e-12:
        return False
    crosses: list[float] = []
    for index in range(4):
        a, b, c = points[index], points[(index + 1) % 4], points[(index + 2) % 4]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) <= 1e-12:
            return False
        crosses.append(cross)
    return all(value > 0 for value in crosses) or all(value < 0 for value in crosses)


def _row_error(task_key: str, path: str, line: int, message: str) -> dict[str, Any]:
    return error(f"invalid_yolo_{task_key}_row", message, path=path, line=line)
