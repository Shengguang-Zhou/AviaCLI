from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any, TypeAlias

import cv2
import numpy as np
from pycocotools import mask as mask_utils

from avia_cli.core.uploads.manifest import _is_image_path
from avia_cli.core.uploads.validation_common import (
    error,
    finite_numbers,
    image_size,
    is_cache_path,
    is_document_path,
)

CocoImageIndex: TypeAlias = tuple[dict[str, Path], dict[str, list[tuple[str, Path]]]]
Taxonomy: TypeAlias = tuple[tuple[object, ...], ...]


def validate_coco(*, source_root: Path, task_key: str) -> tuple[list[str], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    annotation_paths = sorted((source_root / "annotations").glob("*.json"))
    if not annotation_paths:
        return [], [error("missing_coco_annotations", "COCO annotations/*.json is required")]
    image_index = _build_image_index(source_root)
    expected_taxonomy: Taxonomy | None = None
    class_by_id: dict[int, str] = {}
    referenced_images: dict[str, str] = {}
    for path in annotation_paths:
        taxonomy = _validate_annotation_file(
            source_root=source_root,
            path=path,
            task_key=task_key,
            image_index=image_index,
            referenced_images=referenced_images,
            errors=errors,
        )
        if taxonomy is None:
            continue
        if expected_taxonomy is None:
            expected_taxonomy = taxonomy
            class_by_id = {int(item[0]): str(item[1]) for item in taxonomy}
        elif taxonomy != expected_taxonomy:
            errors.append(
                error(
                    "conflicting_coco_taxonomy",
                    "every COCO annotation file must declare one exact category taxonomy",
                    path=path.relative_to(source_root).as_posix(),
                )
            )

    all_images = set(image_index[0])
    for relative in sorted(all_images - set(referenced_images)):
        errors.append(
            error(
                "orphan_coco_image",
                "every COCO image must be referenced by exactly one annotation split",
                path=relative,
            )
        )
    allowed_json = {path.relative_to(source_root).as_posix() for path in annotation_paths}
    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or ".avia" in path.relative_to(source_root).parts:
            continue
        relative = path.relative_to(source_root).as_posix()
        allowed = relative in all_images or relative in allowed_json or is_document_path(relative)
        if not allowed or is_cache_path(relative):
            errors.append(
                error(
                    "unexpected_coco_member",
                    "COCO datasets may contain only referenced images, annotations/*.json, and explicit documentation",
                    path=relative,
                )
            )
    return [class_by_id[key] for key in sorted(class_by_id)], errors


def _validate_annotation_file(
    *,
    source_root: Path,
    path: Path,
    task_key: str,
    image_index: CocoImageIndex,
    referenced_images: dict[str, str],
    errors: list[dict[str, Any]],
) -> Taxonomy | None:
    relative = path.relative_to(source_root).as_posix()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(
            error(
                "invalid_coco_json",
                "COCO annotation JSON is malformed",
                path=relative,
                line=exc.lineno,
                column=exc.colno,
            )
        )
        return None
    except (OSError, UnicodeError) as exc:
        errors.append(error("coco_annotation_read_failed", str(exc), path=relative))
        return None
    if not isinstance(payload, dict):
        errors.append(
            error("invalid_coco_document", "COCO document must be an object", path=relative)
        )
        return None
    categories = payload.get("categories")
    images = payload.get("images")
    annotations = payload.get("annotations")
    if not isinstance(categories, list) or not categories:
        errors.append(
            error("missing_coco_categories", "COCO categories must be non-empty", path=relative)
        )
        return None
    if not isinstance(images, list) or not images:
        errors.append(error("missing_coco_images", "COCO images must be non-empty", path=relative))
        return None
    if not isinstance(annotations, list):
        errors.append(
            error("missing_coco_annotations", "COCO annotations must be an array", path=relative)
        )
        return None

    taxonomy, category_by_id = _categories(
        categories, task_key=task_key, path=relative, errors=errors
    )
    image_by_id = _images(
        images,
        image_index=image_index,
        annotation_path=relative,
        referenced_images=referenced_images,
        errors=errors,
    )
    seen_annotation_ids: set[int] = set()
    for index, raw in enumerate(annotations):
        location = {"path": relative, "annotation_index": index}
        if not isinstance(raw, dict):
            errors.append(
                error("invalid_coco_annotation", "annotation must be an object", **location)
            )
            continue
        annotation_id = _strict_int(raw.get("id"))
        image_id = _strict_int(raw.get("image_id"))
        category_id = _strict_int(raw.get("category_id"))
        if annotation_id is None or annotation_id in seen_annotation_ids:
            errors.append(
                error(
                    "invalid_coco_annotation_id",
                    "annotation id must be a unique integer",
                    **location,
                )
            )
        else:
            seen_annotation_ids.add(annotation_id)
        image_record = image_by_id.get(image_id) if image_id is not None else None
        if image_record is None:
            errors.append(
                error("unknown_coco_image", "annotation references an unknown image", **location)
            )
            continue
        category = category_by_id.get(category_id) if category_id is not None else None
        if category is None:
            errors.append(
                error(
                    "unknown_coco_category", "annotation references an unknown category", **location
                )
            )
            continue
        width, height, _image_relative = image_record
        _validate_bbox(
            raw.get("bbox"), width=width, height=height, location=location, errors=errors
        )
        if task_key == "segment":
            _validate_segmentation(
                raw.get("segmentation"),
                width=width,
                height=height,
                location=location,
                errors=errors,
            )
        elif task_key == "pose":
            _validate_keypoints(
                raw, category=category, width=width, height=height, location=location, errors=errors
            )
    return taxonomy


def _categories(
    categories: list[object], *, task_key: str, path: str, errors: list[dict[str, Any]]
) -> tuple[Taxonomy, dict[int, dict[str, object]]]:
    canonical: list[tuple[object, ...]] = []
    result: dict[int, dict[str, object]] = {}
    names: set[str] = set()
    for index, raw in enumerate(categories):
        location = {"path": path, "category_index": index}
        if not isinstance(raw, dict):
            errors.append(error("invalid_coco_category", "category must be an object", **location))
            continue
        category_id = _strict_int(raw.get("id"))
        name = raw.get("name")
        if (
            category_id is None
            or category_id < 0
            or not isinstance(name, str)
            or not name
            or name != name.strip()
            or category_id in result
            or name in names
        ):
            errors.append(
                error(
                    "invalid_coco_category",
                    "category ids and canonical names must be unique",
                    **location,
                )
            )
            continue
        supercategory = raw.get("supercategory", "")
        if not isinstance(supercategory, str) or supercategory != supercategory.strip():
            errors.append(
                error(
                    "invalid_coco_category", "supercategory must be a canonical string", **location
                )
            )
            continue
        keypoints: tuple[str, ...] = ()
        skeleton: tuple[tuple[int, int], ...] = ()
        if task_key == "pose":
            raw_keypoints = raw.get("keypoints")
            raw_skeleton = raw.get("skeleton")
            if (
                not isinstance(raw_keypoints, list)
                or not raw_keypoints
                or any(
                    not isinstance(item, str) or not item or item != item.strip()
                    for item in raw_keypoints
                )
                or len(set(raw_keypoints)) != len(raw_keypoints)
                or not isinstance(raw_skeleton, list)
            ):
                errors.append(
                    error(
                        "invalid_coco_pose_category",
                        "pose categories require unique keypoint names and a skeleton array",
                        **location,
                    )
                )
                continue
            keypoints = tuple(raw_keypoints)
            skeleton_rows: list[tuple[int, int]] = []
            for edge in raw_skeleton:
                if (
                    not isinstance(edge, list)
                    or len(edge) != 2
                    or any(_strict_int(item) is None for item in edge)
                    or int(edge[0]) < 1
                    or int(edge[1]) < 1
                    or int(edge[0]) > len(keypoints)
                    or int(edge[1]) > len(keypoints)
                    or edge[0] == edge[1]
                ):
                    errors.append(
                        error(
                            "invalid_coco_pose_category",
                            "skeleton edges must reference distinct 1-based keypoints",
                            **location,
                        )
                    )
                    skeleton_rows = []
                    break
                skeleton_rows.append((int(edge[0]), int(edge[1])))
            if raw_skeleton and not skeleton_rows:
                continue
            if len(set(skeleton_rows)) != len(skeleton_rows):
                errors.append(
                    error("invalid_coco_pose_category", "skeleton edges must be unique", **location)
                )
                continue
            skeleton = tuple(skeleton_rows)
        normalized = dict(raw)
        normalized["keypoints"] = list(keypoints)
        result[category_id] = normalized
        names.add(name)
        canonical.append((category_id, name, supercategory, keypoints, skeleton))
    return tuple(sorted(canonical)), result


def _images(
    images: list[object],
    *,
    image_index: CocoImageIndex,
    annotation_path: str,
    referenced_images: dict[str, str],
    errors: list[dict[str, Any]],
) -> dict[int, tuple[int, int, str]]:
    result: dict[int, tuple[int, int, str]] = {}
    for index, raw in enumerate(images):
        location = {"path": annotation_path, "image_index": index}
        if not isinstance(raw, dict):
            errors.append(error("invalid_coco_image", "image must be an object", **location))
            continue
        image_id = _strict_int(raw.get("id"))
        file_name = raw.get("file_name")
        declared_width = _strict_int(raw.get("width"))
        declared_height = _strict_int(raw.get("height"))
        if image_id is None or image_id in result:
            errors.append(
                error("invalid_coco_image_id", "image id must be a unique integer", **location)
            )
            continue
        if not isinstance(file_name, str) or not _canonical_coco_path(file_name):
            errors.append(
                error(
                    "invalid_coco_file_name",
                    "image file_name must be a canonical relative POSIX path",
                    **location,
                )
            )
            continue
        try:
            image_path = _resolve_image_path(image_index, Path(file_name))
            actual = image_size(image_path)
        except (OSError, ValueError) as exc:
            errors.append(error("invalid_image", str(exc), file_name=file_name, **location))
            continue
        resolved_relative = next(
            key for key, value in image_index[0].items() if value == image_path
        )
        previous = referenced_images.get(resolved_relative)
        if previous is not None:
            errors.append(
                error(
                    "duplicate_coco_image_split",
                    "an image may belong to exactly one COCO annotation split",
                    file_name=resolved_relative,
                    first_annotation_path=previous,
                    **location,
                )
            )
        else:
            referenced_images[resolved_relative] = annotation_path
        if declared_width != actual[0] or declared_height != actual[1]:
            errors.append(
                error(
                    "coco_image_size_mismatch",
                    "declared image dimensions do not match the file",
                    file_name=file_name,
                    **location,
                )
            )
            continue
        result[image_id] = (actual[0], actual[1], resolved_relative)
    return result


def _canonical_coco_path(value: str) -> bool:
    return (
        bool(value)
        and value == value.strip()
        and value == unicodedata.normalize("NFC", value)
        and "\\" not in value
        and not value.startswith("/")
        and all(part not in {"", ".", ".."} and part == part.strip() for part in value.split("/"))
        and not any(unicodedata.category(character) == "Cc" for character in value)
    )


def _build_image_index(source_root: Path) -> CocoImageIndex:
    by_relative: dict[str, Path] = {}
    by_name: dict[str, list[tuple[str, Path]]] = {}
    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or not _is_image_path(path.name):
            continue
        relative = path.relative_to(source_root).as_posix()
        by_relative[relative] = path
        by_name.setdefault(path.name, []).append((relative, path))
    return by_relative, by_name


def _resolve_image_path(image_index: CocoImageIndex, relative: Path) -> Path:
    by_relative, by_name = image_index
    suffix = relative.as_posix()
    direct = by_relative.get(suffix)
    if direct is not None:
        return direct
    candidates = [
        path for indexed, path in by_name.get(relative.name, []) if indexed.endswith(f"/{suffix}")
    ]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise ValueError(f"COCO image file does not exist: {suffix}")
    raise ValueError(f"COCO image file_name is ambiguous: {suffix}")


def _validate_bbox(
    value: object,
    *,
    width: int,
    height: int,
    location: dict[str, object],
    errors: list[dict[str, Any]],
) -> None:
    numbers = finite_numbers(value if isinstance(value, list) else [])
    if numbers is None or len(numbers) != 4:
        errors.append(
            error("invalid_coco_bbox", "bbox must contain four finite numbers", **location)
        )
        return
    x, y, box_width, box_height = numbers
    if (
        x < 0
        or y < 0
        or box_width <= 0
        or box_height <= 0
        or x + box_width > width
        or y + box_height > height
    ):
        errors.append(
            error("invalid_coco_bbox", "bbox corners must fit inside the image", **location)
        )


def _validate_segmentation(
    value: object,
    *,
    width: int,
    height: int,
    location: dict[str, object],
    errors: list[dict[str, Any]],
) -> None:
    try:
        mask = _decode_segmentation_mask(value, width=width, height=height)
        _require_lossless_yolo_polygon(mask)
    except (TypeError, ValueError, UnicodeEncodeError, IndexError, OverflowError) as exc:
        errors.append(
            error(
                "invalid_coco_segmentation",
                "segmentation must be one non-empty connected mask without holes and must be exactly representable by one YOLO polygon",
                reason=str(exc),
                **location,
            )
        )


def _decode_segmentation_mask(value: object, *, width: int, height: int) -> np.ndarray:
    if isinstance(value, list) and value:
        polygons: list[object]
        if all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value):
            polygons = [value]
        else:
            polygons = list(value)
        canonical_polygons: list[list[float]] = []
        for polygon in polygons:
            numbers = finite_numbers(polygon if isinstance(polygon, list) else [])
            if numbers is None or len(numbers) < 6 or len(numbers) % 2:
                raise ValueError("COCO polygon segmentation is invalid")
            points = list(zip(numbers[::2], numbers[1::2], strict=True))
            if any(x < 0 or x > width or y < 0 or y > height for x, y in points):
                raise ValueError("COCO polygon coordinates must fit inside the image")
            canonical_polygons.append(numbers)
        rles = mask_utils.frPyObjects(canonical_polygons, height, width)
        decoded = mask_utils.decode(mask_utils.merge(rles))
    elif (
        isinstance(value, dict)
        and set(value) == {"size", "counts"}
        and value.get("size") == [height, width]
    ):
        counts = value.get("counts")
        if isinstance(counts, str):
            rle: object = {
                "size": [height, width],
                "counts": counts.encode("ascii", "strict"),
            }
        elif (
            isinstance(counts, list)
            and counts
            and all(_strict_int(item) is not None and int(item) >= 0 for item in counts)
            and sum(int(item) for item in counts) == height * width
        ):
            rle = mask_utils.frPyObjects({"size": [height, width], "counts": counts}, height, width)
        else:
            raise ValueError("COCO RLE counts are invalid")
        decoded = mask_utils.decode(rle)
    else:
        raise ValueError("COCO segment annotation requires polygon or RLE segmentation")
    mask = np.asarray(decoded, dtype=np.uint8)
    if mask.ndim == 3:
        mask = np.any(mask > 0, axis=2).astype(np.uint8)
    if mask.shape != (height, width) or not np.any(mask):
        raise ValueError("COCO segmentation must be a non-empty image-sized mask")
    return (mask > 0).astype(np.uint8)


def _require_lossless_yolo_polygon(mask: np.ndarray) -> None:
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    hierarchy_rows = [] if hierarchy is None else list(hierarchy[0])
    outer_indices = [index for index, row in enumerate(hierarchy_rows) if int(row[3]) == -1]
    has_holes = any(int(row[3]) != -1 for row in hierarchy_rows)
    if len(outer_indices) != 1 or has_holes:
        raise ValueError("COCO segmentation must be a single connected mask without holes")
    contour = contours[outer_indices[0]].reshape(-1, 2)
    if len(contour) < 3 or len({tuple(point) for point in contour.tolist()}) < 3:
        raise ValueError("COCO segmentation contour has fewer than three unique points")
    rebuilt = np.zeros(mask.shape, dtype=np.uint8)
    cv2.fillPoly(rebuilt, [contour.astype(np.int32)], 1)
    if not np.array_equal(rebuilt, mask):
        raise ValueError("COCO segmentation cannot be represented losslessly by one YOLO polygon")


def _validate_keypoints(
    annotation: dict[str, object],
    *,
    category: dict[str, object],
    width: int,
    height: int,
    location: dict[str, object],
    errors: list[dict[str, Any]],
) -> None:
    names = list(category.get("keypoints") or [])
    numbers = finite_numbers(
        annotation.get("keypoints") if isinstance(annotation.get("keypoints"), list) else []
    )
    if numbers is None or len(numbers) != len(names) * 3:
        errors.append(
            error("invalid_coco_keypoints", "keypoints must match category metadata", **location)
        )
        return
    visible = 0
    for offset in range(0, len(numbers), 3):
        x, y, visibility = numbers[offset : offset + 3]
        invalid = (
            visibility not in {0.0, 1.0, 2.0}
            or (visibility == 0 and (x != 0 or y != 0))
            or (visibility > 0 and (x < 0 or x > width or y < 0 or y > height))
        )
        if invalid:
            errors.append(
                error(
                    "invalid_coco_keypoints",
                    "visible keypoints must fit the image and invisible keypoints must be 0,0,0",
                    **location,
                )
            )
            return
        visible += int(visibility > 0)
    if _strict_int(annotation.get("num_keypoints")) != visible:
        errors.append(
            error(
                "invalid_coco_num_keypoints",
                "num_keypoints does not match visible keypoints",
                **location,
            )
        )


def _strict_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
