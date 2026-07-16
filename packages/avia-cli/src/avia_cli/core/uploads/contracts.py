from __future__ import annotations

import unicodedata

FORMAT_TASKS: dict[str, frozenset[str]] = {
    "yolo": frozenset({"detect", "classify", "segment", "pose", "obb"}),
    "coco": frozenset({"detect", "segment", "pose"}),
    "imagenet": frozenset({"classify"}),
    "anomalib": frozenset({"ad"}),
}


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
