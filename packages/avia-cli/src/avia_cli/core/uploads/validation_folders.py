from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from avia_cli.core.uploads.manifest import _is_image_path, is_client_state_path
from avia_cli.core.uploads.validation_common import (
    dataset_role_directories,
    error,
    image_files,
    image_size,
    is_cache_path,
    is_document_path,
)


def validate_imagenet(source_root: Path) -> tuple[list[str], list[dict[str, Any]]]:
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
            paths = image_files(source_root=source_root, root=split_root / class_name)
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
        allowed=lambda path: (
            len(Path(path).parts) >= 3
            and Path(path).parts[0] in {"train", "val", "test"}
            and Path(path).parts[1] in expected
            and _is_image_path(path)
        ),
        code="unexpected_imagenet_member",
        errors=errors,
    )
    return classes, errors


def validate_anomalib(source_root: Path) -> tuple[list[str], list[dict[str, Any]]]:
    errors: list[dict[str, Any]] = []
    train_good = image_files(source_root=source_root, root=source_root / "train" / "good")
    test_good = image_files(source_root=source_root, root=source_root / "test" / "good")
    validation_good = image_files(
        source_root=source_root,
        root=source_root / "validation" / "good",
    )
    if not train_good:
        errors.append(
            error("missing_anomalib_train_good", "Anomalib train/good images are required")
        )
    if not test_good and not validation_good:
        errors.append(
            error(
                "missing_anomalib_good_evaluation",
                "Anomalib test/good or validation/good images are required",
            )
        )
    test_root = source_root / "test"
    defect_dirs = [
        path
        for path in dataset_role_directories(source_root=source_root, role_root=test_root)
        if path.name != "good"
    ]
    bad_samples = [
        (defect.name, sample)
        for defect in defect_dirs
        for sample in image_files(source_root=source_root, root=defect)
    ]
    if not bad_samples:
        errors.append(
            error("missing_anomalib_bad_samples", "Anomalib test/<defect> images are required")
        )

    all_paths = [
        *train_good,
        *test_good,
        *validation_good,
        *(path for _defect, path in bad_samples),
    ]
    sizes = _validate_images(source_root, all_paths, errors)
    expected_masks: set[Path] = set()
    for defect, sample in bad_samples:
        expected_stem = sample.stem if defect == "bad" else f"{sample.stem}_mask"
        candidates = sorted(
            path
            for path in image_files(
                source_root=source_root,
                root=source_root / "ground_truth" / defect,
            )
            if path.stem == expected_stem
        )
        if len(candidates) != 1:
            errors.append(
                error(
                    "missing_anomaly_mask",
                    "every bad sample must have exactly one ground_truth mask",
                    path=sample.relative_to(source_root).as_posix(),
                )
            )
            continue
        mask = candidates[0]
        expected_masks.add(mask.resolve())
        mask_sizes = _validate_images(source_root, [mask], errors)
        sample_size = sizes.get(sample.resolve())
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
    actual_masks = {
        path.resolve()
        for path in image_files(source_root=source_root, root=source_root / "ground_truth")
    }
    for orphan in sorted(actual_masks - expected_masks):
        errors.append(
            error(
                "orphan_anomaly_mask",
                "ground_truth mask has no matching bad sample",
                path=orphan.relative_to(source_root.resolve()).as_posix(),
            )
        )
    allowed_images = {
        path.resolve().relative_to(source_root.resolve()).as_posix()
        for path in [*all_paths, *actual_masks]
    }
    _validate_folder_roles(
        source_root,
        allowed=lambda path: path in allowed_images,
        code="unexpected_anomalib_member",
        errors=errors,
    )
    return ["good", *sorted({defect for defect, _sample in bad_samples})], errors


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
    allowed: Any,
    code: str,
    errors: list[dict[str, Any]],
) -> None:
    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or is_client_state_path(path.relative_to(source_root)):
            continue
        relative = path.relative_to(source_root).as_posix()
        if (not allowed(relative) and not is_document_path(relative)) or is_cache_path(relative):
            errors.append(
                error(
                    code,
                    "dataset contains a file outside the exact task layout",
                    path=relative,
                )
            )
