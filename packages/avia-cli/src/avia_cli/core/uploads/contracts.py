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


def require_format_task(*, format_name: str, task_key: str) -> tuple[str, str]:
    exact_format = str(format_name)
    exact_task = str(task_key)
    supported_tasks = FORMAT_TASKS.get(exact_format)
    if supported_tasks is None or exact_task not in supported_tasks:
        raise SystemExit(f"format '{exact_format}' does not support task '{exact_task}'")
    return exact_format, exact_task


def require_object_prefix_uri(uri: object) -> str:
    value = str(uri)
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
