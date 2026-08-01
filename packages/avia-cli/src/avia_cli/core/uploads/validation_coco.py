from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import cv2
import numpy as np
from pycocotools import mask as mask_utils

from avia_cli.core.uploads.inventory import DatasetRoleInventory
from avia_cli.core.uploads.manifest import is_client_state_path
from avia_cli.core.strict_json import strict_json_loads
from avia_cli.core.uploads.validation_common import (
    error,
    image_size,
    is_cache_path,
    is_document_path,
    json_finite_numbers,
)

CocoImageIndex: TypeAlias = dict[str, Path]
TaxonomyIdentity: TypeAlias = str


@dataclass(frozen=True, slots=True)
class _CocoAnnotationDocument:
    path: Path
    payload: dict[str, Any]
    category_by_id: dict[int, dict[str, object]]


def validate_coco(
    *,
    source_root: Path,
    inventory: DatasetRoleInventory,
    task_key: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    classes, documents = _parse_coco_catalog(
        source_root=source_root,
        inventory=inventory,
        task_key=task_key,
        errors=errors,
    )
    image_index = _build_image_index(source_root, inventory=inventory)
    referenced_images: dict[str, str] = {}
    for document in documents:
        _validate_annotation_records(
            source_root=source_root,
            document=document,
            task_key=task_key,
            image_index=image_index,
            referenced_images=referenced_images,
            errors=errors,
        )

    all_images = set(image_index)
    for relative in sorted(all_images - set(referenced_images)):
        errors.append(
            error(
                "orphan_coco_image",
                "every COCO image must be referenced by exactly one annotation split",
                path=relative,
            )
        )
    allowed_json = set(inventory.label_paths)
    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or is_client_state_path(path.relative_to(source_root)):
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
    return classes, errors


def inspect_coco_class_catalog(
    *,
    source_root: Path,
    inventory: DatasetRoleInventory,
    task_key: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    classes, _documents = _parse_coco_catalog(
        source_root=source_root,
        inventory=inventory,
        task_key=task_key,
        errors=errors,
    )
    return classes, errors


def _parse_coco_catalog(
    *,
    source_root: Path,
    inventory: DatasetRoleInventory,
    task_key: str,
    errors: list[dict[str, Any]],
) -> tuple[list[str], list[_CocoAnnotationDocument]]:
    annotation_paths = [source_root / relative for relative in inventory.label_paths]
    if not annotation_paths:
        errors.append(error("missing_coco_annotations", "COCO annotations/*.json is required"))
        return [], []

    expected_taxonomy: TaxonomyIdentity | None = None
    class_by_id: dict[int, str] = {}
    documents: list[_CocoAnnotationDocument] = []
    for path in annotation_paths:
        payload = _read_annotation_document(
            source_root=source_root,
            path=path,
            errors=errors,
        )
        if payload is None:
            continue
        relative = path.relative_to(source_root).as_posix()
        categories: list[object] = payload["categories"]
        taxonomy, category_by_id = _categories(
            categories,
            task_key=task_key,
            path=relative,
            errors=errors,
        )
        if expected_taxonomy is None:
            expected_taxonomy = taxonomy
            class_by_id = {
                category_id: str(category["name"])
                for category_id, category in category_by_id.items()
            }
        elif taxonomy != expected_taxonomy:
            errors.append(
                error(
                    "conflicting_coco_taxonomy",
                    "every COCO annotation file must declare one exact category taxonomy",
                    path=relative,
                )
            )
        documents.append(
            _CocoAnnotationDocument(
                path=path,
                payload=payload,
                category_by_id=category_by_id,
            )
        )
    return [class_by_id[key] for key in sorted(class_by_id)], documents


def _read_annotation_document(
    *,
    source_root: Path,
    path: Path,
    errors: list[dict[str, Any]],
) -> dict[str, Any] | None:
    relative = path.relative_to(source_root).as_posix()
    try:
        payload = strict_json_loads(path.read_text(encoding="utf-8"))
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
    except ValueError as exc:
        errors.append(
            error(
                "invalid_coco_json",
                "COCO annotation JSON violates the strict JSON contract",
                path=relative,
                reason=str(exc),
            )
        )
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

    return payload


def _validate_annotation_records(
    *,
    source_root: Path,
    document: _CocoAnnotationDocument,
    task_key: str,
    image_index: CocoImageIndex,
    referenced_images: dict[str, str],
    errors: list[dict[str, Any]],
) -> None:
    relative = document.path.relative_to(source_root).as_posix()
    images: list[object] = document.payload["images"]
    annotations: list[object] = document.payload["annotations"]

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
        if annotation_id is None or annotation_id < 0 or annotation_id in seen_annotation_ids:
            errors.append(
                error(
                    "invalid_coco_annotation_id",
                    "annotation id must be a unique non-negative integer",
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
        category = document.category_by_id.get(category_id) if category_id is not None else None
        if category is None:
            errors.append(
                error(
                    "unknown_coco_category", "annotation references an unknown category", **location
                )
            )
            continue
        width, height, _image_relative = image_record
        _validate_annotation_record(raw, location=location, errors=errors)
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


def _categories(
    categories: list[object], *, task_key: str, path: str, errors: list[dict[str, Any]]
) -> tuple[TaxonomyIdentity, dict[int, dict[str, object]]]:
    pose_schemas: set[tuple[tuple[str, ...], tuple[tuple[int, int], ...]]] = set()
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
        supercategory = raw.get("supercategory")
        if "supercategory" in raw and (
            not isinstance(supercategory, str) or supercategory != supercategory.strip()
        ):
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
            raw_skeleton = raw.get("skeleton", [])
            if (
                not isinstance(raw_keypoints, list)
                or not raw_keypoints
                or len(raw_keypoints) > 2048
                or any(
                    not isinstance(item, str)
                    or not item
                    or item != item.strip()
                    or len(item) > 200
                    or any(unicodedata.category(character) == "Cc" for character in item)
                    for item in raw_keypoints
                )
                or len(set(raw_keypoints)) != len(raw_keypoints)
                or not isinstance(raw_skeleton, list)
            ):
                errors.append(
                    error(
                        "invalid_coco_pose_category",
                        "pose categories require unique keypoint names and an optional skeleton array",
                        **location,
                    )
                )
                continue
            keypoints = tuple(raw_keypoints)
            skeleton_rows: list[tuple[int, int]] = []
            seen_skeleton_edges: set[tuple[int, int]] = set()
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
                skeleton_edge = (int(edge[0]), int(edge[1]))
                canonical_edge = (
                    min(skeleton_edge[0], skeleton_edge[1]),
                    max(skeleton_edge[0], skeleton_edge[1]),
                )
                if canonical_edge in seen_skeleton_edges:
                    errors.append(
                        error(
                            "invalid_coco_pose_category",
                            "skeleton edges must be unique",
                            **location,
                        )
                    )
                    skeleton_rows = []
                    break
                seen_skeleton_edges.add(canonical_edge)
                skeleton_rows.append(skeleton_edge)
            if raw_skeleton and not skeleton_rows:
                continue
            skeleton = tuple(skeleton_rows)
            missing_counterpart = _missing_pose_counterpart(keypoints)
            if missing_counterpart is not None:
                keypoint, counterpart = missing_counterpart
                errors.append(
                    error(
                        "invalid_coco_pose_category",
                        "left/right keypoint counterpart is missing",
                        keypoint=keypoint,
                        counterpart=counterpart,
                        **location,
                    )
                )
                continue
        normalized = dict(raw)
        if task_key == "pose":
            normalized["keypoints"] = list(keypoints)
            normalized["skeleton"] = [list(edge) for edge in skeleton]
            pose_schemas.add((keypoints, skeleton))
        result[category_id] = normalized
        names.add(name)
    if task_key == "pose" and len(pose_schemas) > 1:
        errors.append(
            error(
                "conflicting_coco_pose_schema",
                "all COCO categories must declare the same pose keypoint schema",
                path=path,
            )
        )
    return _canonical_json_identity([result[category_id] for category_id in sorted(result)]), result


def _canonical_json_identity(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _missing_pose_counterpart(names: tuple[str, ...]) -> tuple[str, str] | None:
    known = set(names)
    for name in names:
        counterpart: str | None = None
        if name == "left":
            counterpart = "right"
        elif name == "right":
            counterpart = "left"
        elif name.startswith("left_"):
            counterpart = f"right_{name.removeprefix('left_')}"
        elif name.startswith("right_"):
            counterpart = f"left_{name.removeprefix('right_')}"
        if counterpart is not None and counterpart not in known:
            return name, counterpart
    return None


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
        resolved_relative = file_name
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


def _build_image_index(
    source_root: Path,
    *,
    inventory: DatasetRoleInventory,
) -> CocoImageIndex:
    return {relative: source_root / relative for relative in inventory.image_paths}


def _resolve_image_path(image_index: CocoImageIndex, relative: Path) -> Path:
    suffix = relative.as_posix()
    direct = image_index.get(suffix)
    if direct is not None:
        return direct
    raise ValueError(f"COCO image file does not exist at exact path: {suffix}")


def _validate_bbox(
    value: object,
    *,
    width: int,
    height: int,
    location: dict[str, object],
    errors: list[dict[str, Any]],
) -> None:
    numbers = json_finite_numbers(value)
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


def _validate_annotation_record(
    annotation: dict[str, object],
    *,
    location: dict[str, object],
    errors: list[dict[str, Any]],
) -> None:
    area = json_finite_numbers([annotation.get("area")])
    if area is None or area[0] <= 0:
        errors.append(
            error(
                "invalid_coco_area", "annotation area must be a positive finite number", **location
            )
        )
    if _strict_int(annotation.get("iscrowd")) not in {0, 1}:
        errors.append(
            error("invalid_coco_iscrowd", "annotation iscrowd must be exactly 0 or 1", **location)
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
            numbers = json_finite_numbers(polygon)
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
        elif isinstance(counts, list):
            canonical_counts = _validate_uncompressed_rle_counts(
                counts,
                height=height,
                width=width,
            )
            rle = mask_utils.frPyObjects(
                {"size": [height, width], "counts": canonical_counts},
                height,
                width,
            )
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


def _validate_uncompressed_rle_counts(
    counts: list[object],
    *,
    height: int,
    width: int,
) -> list[int]:
    if not counts:
        raise ValueError("COCO RLE counts are invalid")
    canonical: list[int] = []
    for raw_count in counts:
        count = _strict_int(raw_count)
        if count is None or count < 0:
            raise ValueError("COCO RLE counts are invalid")
        canonical.append(count)
    if sum(canonical) != height * width:
        raise ValueError("COCO RLE counts must cover the exact image area")
    if any(count == 0 for count in canonical[1:]):
        raise ValueError("COCO RLE permits a zero run only as the first count")
    return canonical


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
    numbers = json_finite_numbers(annotation.get("keypoints"))
    if numbers is None or len(numbers) != len(names) * 3:
        errors.append(
            error("invalid_coco_keypoints", "keypoints must match category metadata", **location)
        )
        return
    visible = 0
    for offset in range(0, len(numbers), 3):
        x, y, visibility = numbers[offset : offset + 3]
        invalid = visibility not in {0.0, 1.0, 2.0} or x < 0 or x > width or y < 0 or y > height
        if invalid:
            errors.append(
                error(
                    "invalid_coco_keypoints",
                    "keypoints must use visibility 0, 1, or 2 and fit inside the image",
                    **location,
                )
            )
            return
        visible += int(visibility > 0)
    if "num_keypoints" in annotation and _strict_int(annotation.get("num_keypoints")) != visible:
        errors.append(
            error(
                "invalid_coco_num_keypoints",
                "num_keypoints does not match visible keypoints",
                **location,
            )
        )


def _strict_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
