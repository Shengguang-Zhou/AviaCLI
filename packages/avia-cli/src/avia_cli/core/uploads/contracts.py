from __future__ import annotations

import unicodedata

from avia_cli.core.uploads.class_catalog import require_canonical_class_catalog

FORMAT_TASKS: dict[str, frozenset[str]] = {
    "yolo": frozenset({"detect", "classify", "segment", "pose", "obb"}),
    "coco": frozenset({"detect", "segment", "pose"}),
    "imagenet": frozenset({"classify"}),
    "anomalib": frozenset({"ad"}),
}
ANOMALIB_CLASSES = ("good", "bad")
SUPPORTED_FORMATS = tuple(FORMAT_TASKS)
SUPPORTED_TASK_KEYS = ("detect", "classify", "segment", "pose", "obb", "ad")


def require_object_prefix_class_catalog(
    value: object,
    *,
    format_name: str,
    label: str,
) -> list[str]:
    classes = require_canonical_class_catalog(
        value,
        label=label,
        allow_empty=True,
    )
    if format_name == "anomalib":
        if classes != list(ANOMALIB_CLASSES):
            raise ValueError(f"{label} must be exactly ['good', 'bad'] for anomalib")
    elif format_name in {"coco", "imagenet"}:
        if classes:
            raise ValueError(f"{label} must be empty for {format_name}")
    elif format_name != "yolo":
        raise ValueError(f"{label} format is unsupported: {format_name}")
    return classes


def require_folder_class_catalog(
    value: object,
    *,
    format_name: str,
    label: str,
) -> list[str]:
    classes = require_canonical_class_catalog(value, label=label)
    if format_name == "anomalib" and classes != list(ANOMALIB_CLASSES):
        raise ValueError(f"{label} must be exactly ['good', 'bad'] for anomalib")
    if format_name not in FORMAT_TASKS:
        raise ValueError(f"{label} format is unsupported: {format_name}")
    return classes


def require_format_task(*, format_name: object, task_key: object) -> tuple[str, str]:
    if not isinstance(format_name, str) or not isinstance(task_key, str):
        raise SystemExit("format and task must be strings")
    supported_tasks = FORMAT_TASKS.get(format_name)
    if supported_tasks is None or task_key not in supported_tasks:
        raise SystemExit(f"format '{format_name}' does not support task '{task_key}'")
    return format_name, task_key


def require_object_prefix_uri(uri: object) -> str:
    if not isinstance(uri, str):
        raise SystemExit("object-prefix URI must be a string")
    value = uri
    if (
        not value
        or value != value.strip()
        or value.startswith(("/", "s3://"))
        or not value.endswith("/")
        or "//" in value
        or "\\" in value
        or value != unicodedata.normalize("NFC", value)
        or any(unicodedata.category(character) == "Cc" for character in value)
        or any(part in {"", ".", ".."} or part != part.strip() for part in value[:-1].split("/"))
    ):
        raise SystemExit(
            "object-prefix URI must be a canonical bare object path ending in '/': "
            "for example datasets/project/"
        )
    return value
