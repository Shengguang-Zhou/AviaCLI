from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from avia_cli.core.uploads.class_catalog import require_canonical_class_catalog
from avia_cli.core.uploads.contracts import ANOMALIB_CLASSES
from avia_cli.core.uploads.inventory import require_manifest_inventory
from avia_cli.core.uploads.validation_coco import inspect_coco_class_catalog, validate_coco
from avia_cli.core.uploads.validation_common import error
from avia_cli.core.uploads.validation_folders import (
    inspect_imagenet_class_catalog,
    validate_anomalib,
    validate_imagenet,
)
from avia_cli.core.uploads.validation_yolo import validate_yolo


def validate_dataset(
    *,
    source_root: Path,
    manifest: dict[str, object],
    format_name: str,
    task_key: str,
    declared_classes: list[str] | None = None,
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    inventory = require_manifest_inventory(manifest, format_name=format_name)
    if format_name == "yolo":
        classes, errors, warnings = validate_yolo(
            source_root=source_root,
            manifest=manifest,
            inventory=inventory,
            task_key=task_key,
            declared_classes=declared_classes,
        )
    elif format_name == "coco":
        classes, errors = validate_coco(
            source_root=source_root,
            inventory=inventory,
            task_key=task_key,
        )
        warnings = []
    elif format_name == "imagenet":
        classes, errors = validate_imagenet(source_root, inventory=inventory)
        warnings = []
    elif format_name == "anomalib":
        classes, errors = validate_anomalib(source_root, inventory=inventory)
        warnings = []
    else:
        raise AssertionError(f"unreachable dataset format: {format_name}")
    classes = _canonical_dataset_class_catalog(
        classes,
        format_name=format_name,
        errors=errors,
    )
    return classes, errors, warnings


def inspect_dataset_class_catalog(
    *,
    source_root: Path,
    manifest: dict[str, object],
    format_name: str,
    task_key: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    inventory = require_manifest_inventory(manifest, format_name=format_name)
    errors: list[dict[str, Any]] = []
    classes: object
    if format_name == "yolo":
        classes = manifest.get("classes")
    elif format_name == "coco":
        classes, errors = inspect_coco_class_catalog(
            source_root=source_root,
            inventory=inventory,
            task_key=task_key,
        )
    elif format_name == "imagenet":
        classes, errors = inspect_imagenet_class_catalog(
            source_root,
            inventory=inventory,
        )
    elif format_name == "anomalib":
        classes = list(ANOMALIB_CLASSES)
    else:
        raise AssertionError(f"unreachable dataset format: {format_name}")
    return (
        _canonical_dataset_class_catalog(
            classes,
            format_name=format_name,
            errors=errors,
        ),
        errors,
    )


def _canonical_dataset_class_catalog(
    classes: object,
    *,
    format_name: str,
    errors: list[dict[str, Any]],
) -> list[str]:
    try:
        return require_canonical_class_catalog(
            classes,
            label=f"{format_name} class catalog",
            allow_empty=bool(errors),
        )
    except ValueError as exc:
        errors.append(error("invalid_class_catalog", str(exc)))
        return []


def require_valid_dataset(
    *,
    source_root: Path,
    manifest: dict[str, object],
    format_name: str,
    task_key: str,
    declared_classes: list[str] | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    classes, errors, warnings = validate_dataset(
        source_root=source_root,
        manifest=manifest,
        format_name=format_name,
        task_key=task_key,
        declared_classes=declared_classes,
    )
    if errors:
        raise SystemExit(
            json.dumps(
                {
                    "message": "dataset validation failed before upload",
                    "format": format_name,
                    "task_key": task_key,
                    "error_count": len(errors),
                    "errors": errors,
                },
                ensure_ascii=False,
            )
        )
    return classes, warnings
