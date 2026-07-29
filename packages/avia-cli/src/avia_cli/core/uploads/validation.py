from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from avia_cli.core.uploads.inventory import require_manifest_inventory
from avia_cli.core.uploads.validation_coco import validate_coco
from avia_cli.core.uploads.validation_folders import validate_anomalib, validate_imagenet
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
        return validate_yolo(
            source_root=source_root,
            manifest=manifest,
            inventory=inventory,
            task_key=task_key,
            declared_classes=declared_classes,
        )
    if format_name == "coco":
        classes, errors = validate_coco(
            source_root=source_root,
            inventory=inventory,
            task_key=task_key,
        )
        return classes, errors, []
    if format_name == "imagenet":
        classes, errors = validate_imagenet(source_root, inventory=inventory)
        return classes, errors, []
    if format_name == "anomalib":
        classes, errors = validate_anomalib(source_root, inventory=inventory)
        return classes, errors, []
    raise AssertionError(f"unreachable dataset format: {format_name}")


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
