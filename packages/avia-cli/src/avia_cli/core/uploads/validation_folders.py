from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from PIL import Image

from avia_cli.core.uploads.contracts import ANOMALIB_CLASSES
from avia_cli.core.uploads.inventory import DatasetRoleInventory
from avia_cli.core.uploads.manifest import is_client_state_path
from avia_cli.core.uploads.validation_common import (
    dataset_role_directories,
    error,
    image_size,
    is_cache_path,
    is_document_path,
)

_ANOMALIB_ROLES = (
    ("train", "good"),
    ("val", "good"),
    ("val", "bad"),
    ("test", "good"),
    ("test", "bad"),
)
_ANOMALIB_EVALUATION_SPLITS = ("val", "test")
_ANOMALIB_ROOT_DOCUMENT_NAMES = frozenset({"README", "LICENSE", "source_records.json"})
_ANOMALIB_DIRECTORIES = frozenset(
    {"ground_truth"}
    | {split for split, _class_name in _ANOMALIB_ROLES}
    | {f"{split}/{class_name}" for split, class_name in _ANOMALIB_ROLES}
    | {
        relative
        for split in _ANOMALIB_EVALUATION_SPLITS
        for relative in (f"ground_truth/{split}", f"ground_truth/{split}/bad")
    }
)


def validate_imagenet(
    source_root: Path,
    *,
    inventory: DatasetRoleInventory,
) -> tuple[list[str], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    train_root = source_root / "train"
    classes = [
        path.name
        for path in dataset_role_directories(source_root=source_root, role_root=train_root)
    ]
    if not classes:
        return [], [
            error(
                "missing_imagenet_train_classes", "ImageNet train/<class> directories are required"
            )
        ]
    expected = set(classes)
    for split in ("train", "val", "test"):
        split_root = source_root / split
        if not split_root.exists():
            continue
        actual = {
            path.name
            for path in dataset_role_directories(
                source_root=source_root,
                role_root=split_root,
            )
        }
        if actual != expected:
            errors.append(
                error(
                    "imagenet_class_mismatch",
                    "every ImageNet split must contain the same class directories",
                    split=split,
                    expected=classes,
                    actual=sorted(actual),
                )
            )
        for class_name in sorted(actual):
            paths = [
                source_root / relative
                for relative in inventory.image_paths
                if Path(relative).parts[:2] == (split, class_name)
            ]
            if not paths:
                errors.append(
                    error(
                        "empty_imagenet_class",
                        "ImageNet class directory has no images",
                        split=split,
                        class_name=class_name,
                    )
                )
            _validate_images(source_root, paths, errors)
    if not (source_root / "val").is_dir() and not (source_root / "test").is_dir():
        errors.append(
            error("missing_imagenet_evaluation_split", "ImageNet val or test split is required")
        )
    _validate_folder_roles(
        source_root,
        allowed=set(inventory.image_paths).__contains__,
        code="unexpected_imagenet_member",
        errors=errors,
    )
    return classes, errors


def validate_anomalib(
    source_root: Path,
    *,
    inventory: DatasetRoleInventory,
) -> tuple[list[str], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    _validate_anomalib_directories(source_root=source_root, errors=errors)

    images_by_role: dict[tuple[str, str], list[Path]] = {}
    for split, class_name in _ANOMALIB_ROLES:
        paths = [
            source_root / relative
            for relative in inventory.image_paths
            if Path(relative).parts[:2] == (split, class_name)
        ]
        images_by_role[(split, class_name)] = paths
        if not paths:
            errors.append(
                error(
                    "empty_anomalib_role",
                    "canonical Anomalib roles must each contain at least one source image",
                    split=split,
                    class_name=class_name,
                    path=f"{split}/{class_name}",
                )
            )

    all_sources = [path for role in _ANOMALIB_ROLES for path in images_by_role[role]]
    _validate_anomalib_source_stems(
        source_root=source_root,
        images_by_role=images_by_role,
        errors=errors,
    )
    source_sizes = _validate_images(source_root, all_sources, errors)

    expected_masks: set[Path] = set()
    matched_masks: set[Path] = set()
    for split in _ANOMALIB_EVALUATION_SPLITS:
        for sample in images_by_role[(split, "bad")]:
            mask = source_root / "ground_truth" / split / "bad" / f"{sample.stem}.png"
            expected_masks.add(mask)
            if not mask.is_file():
                errors.append(
                    error(
                        "missing_anomaly_mask",
                        "every bad evaluation image requires the exact same-stem PNG mask",
                        split=split,
                        path=sample.relative_to(source_root).as_posix(),
                        expected_mask=mask.relative_to(source_root).as_posix(),
                    )
                )
                continue
            if mask in matched_masks:
                continue
            matched_masks.add(mask)
            mask_sizes = _validate_images(source_root, [mask], errors)
            sample_size = source_sizes.get(sample.resolve())
            mask_size = mask_sizes.get(mask.resolve())
            if sample_size is not None and mask_size is not None and sample_size != mask_size:
                errors.append(
                    error(
                        "anomaly_mask_size_mismatch",
                        "anomaly mask dimensions must match the bad sample",
                        path=sample.relative_to(source_root).as_posix(),
                        mask_path=mask.relative_to(source_root).as_posix(),
                        image_size=list(sample_size),
                        mask_size=list(mask_size),
                    )
                )
            _validate_anomaly_mask(source_root=source_root, mask=mask, errors=errors)

    actual_masks = {source_root / relative for relative in inventory.mask_paths}
    for orphan in sorted(actual_masks - expected_masks):
        errors.append(
            error(
                "orphan_anomaly_mask",
                "ground_truth mask has no same-stem bad image in the same split",
                path=orphan.relative_to(source_root).as_posix(),
            )
        )

    allowed_images = set(inventory.image_paths) | {
        path.relative_to(source_root).as_posix() for path in matched_masks
    }
    _validate_folder_roles(
        source_root,
        allowed=lambda path: path in allowed_images,
        document_allowed=_is_anomalib_document_path,
        code="unexpected_anomalib_member",
        errors=errors,
    )
    return list(ANOMALIB_CLASSES), errors


def _validate_anomalib_directories(
    *,
    source_root: Path,
    errors: list[dict[str, Any]],
) -> None:
    actual = {
        path.relative_to(source_root).as_posix()
        for path in source_root.rglob("*")
        if path.is_dir() and not is_client_state_path(path.relative_to(source_root))
    }
    for relative in sorted(_ANOMALIB_DIRECTORIES - actual):
        errors.append(
            error(
                "missing_anomalib_directory",
                "canonical Anomalib directory is missing",
                path=relative,
            )
        )
    for relative in sorted(actual - _ANOMALIB_DIRECTORIES):
        errors.append(
            error(
                "unexpected_anomalib_directory",
                "Anomalib dataset contains a directory outside the canonical layout",
                path=relative,
            )
        )


def _is_anomalib_document_path(relative_path: str) -> bool:
    return "/" not in relative_path and relative_path in _ANOMALIB_ROOT_DOCUMENT_NAMES


def _validate_anomalib_source_stems(
    *,
    source_root: Path,
    images_by_role: dict[tuple[str, str], list[Path]],
    errors: list[dict[str, Any]],
) -> None:
    for (split, class_name), paths in images_by_role.items():
        by_stem: dict[str, list[Path]] = {}
        for path in paths:
            by_stem.setdefault(path.stem, []).append(path)
        for stem, duplicates in sorted(by_stem.items()):
            if len(duplicates) <= 1:
                continue
            errors.append(
                error(
                    "duplicate_anomalib_source_stem",
                    "source image stems must be unique within each Anomalib role",
                    split=split,
                    class_name=class_name,
                    stem=stem,
                    paths=[path.relative_to(source_root).as_posix() for path in duplicates],
                )
            )


def _validate_images(
    source_root: Path, paths: list[Path], errors: list[dict[str, Any]]
) -> dict[Path, tuple[int, int]]:
    sizes: dict[Path, tuple[int, int]] = {}
    for path in paths:
        try:
            sizes[path.resolve()] = image_size(path)
        except (OSError, ValueError) as exc:
            errors.append(
                error("invalid_image", str(exc), path=path.relative_to(source_root).as_posix())
            )
    return sizes


def _validate_anomaly_mask(*, source_root: Path, mask: Path, errors: list[dict[str, Any]]) -> None:
    try:
        with Image.open(mask) as image:
            image.load()
            if len(image.getbands()) != 1:
                raise ValueError(f"mask must be single-channel, got mode={image.mode}")
            values = set(image.get_flattened_data())
    except (OSError, ValueError) as exc:
        errors.append(
            error(
                "invalid_anomaly_mask",
                str(exc),
                path=mask.relative_to(source_root).as_posix(),
            )
        )
        return
    if not values or not values.issubset({0, 255}) or 255 not in values:
        errors.append(
            error(
                "invalid_anomaly_mask",
                "anomaly masks must be non-empty binary single-channel images with values 0 and 255",
                path=mask.relative_to(source_root).as_posix(),
                values=sorted(values)[:16],
            )
        )


def _validate_folder_roles(
    source_root: Path,
    *,
    allowed: Callable[[str], bool],
    code: str,
    errors: list[dict[str, Any]],
    document_allowed: Callable[[str], bool] = is_document_path,
) -> None:
    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or is_client_state_path(path.relative_to(source_root)):
            continue
        relative = path.relative_to(source_root).as_posix()
        if (not allowed(relative) and not document_allowed(relative)) or is_cache_path(relative):
            errors.append(
                error(
                    code,
                    "dataset contains a file outside the exact task layout",
                    path=relative,
                )
            )
