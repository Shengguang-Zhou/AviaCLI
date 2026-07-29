from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

from avia_cli.core.uploads.contracts import FORMAT_TASKS

DATASET_IMAGE_SUFFIXES = frozenset({".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"})
_ANOMALIB_IMAGE_SUFFIXES = frozenset({".jpeg", ".jpg", ".png", ".webp"})

_ANOMALIB_SOURCE_ROLES = frozenset(
    {
        ("train", "good"),
        ("val", "good"),
        ("val", "bad"),
        ("test", "good"),
        ("test", "bad"),
    }
)


def is_dataset_image_path(path: str) -> bool:
    """Return whether a member must be decoded as an image.

    This predicate is intentionally about the file encoding only. Product
    counts come from :func:`build_role_inventory`, which also understands the
    selected dataset format and the member's canonical role.
    """

    return Path(path).suffix.lower() in DATASET_IMAGE_SUFFIXES


def requires_lowercase_media_suffix(path: str, *, format_name: str) -> bool:
    if format_name != "anomalib":
        return False
    member = PurePosixPath(path)
    parts = member.parts
    suffix = member.suffix.lower()
    return (_is_anomalib_source_role(parts) and suffix in _ANOMALIB_IMAGE_SUFFIXES) or (
        _is_anomalib_mask_role(parts) and suffix == ".png"
    )


@dataclass(frozen=True, slots=True)
class DatasetRoleInventory:
    format_name: str
    image_paths: tuple[str, ...]
    label_paths: tuple[str, ...]
    mask_paths: tuple[str, ...]

    @property
    def image_count(self) -> int:
        return len(self.image_paths)

    @property
    def label_count(self) -> int:
        return len(self.label_paths)

    @property
    def mask_count(self) -> int:
        return len(self.mask_paths)

    def counts(self) -> dict[str, int]:
        return {
            "image_count": self.image_count,
            "label_count": self.label_count,
            "mask_count": self.mask_count,
        }


def build_role_inventory(
    relative_paths: Iterable[str],
    *,
    format_name: str,
) -> DatasetRoleInventory:
    if format_name not in FORMAT_TASKS:
        raise RuntimeError(f"unsupported dataset inventory format: {format_name!r}")

    images: list[str] = []
    labels: list[str] = []
    masks: list[str] = []
    seen: set[str] = set()
    for value in relative_paths:
        if not isinstance(value, str) or not value:
            raise RuntimeError("dataset manifest relative_path must be a non-empty string")
        if value in seen:
            raise RuntimeError(f"dataset manifest contains duplicate member: {value}")
        seen.add(value)
        role = _member_role(value, format_name=format_name)
        if role == "image":
            images.append(value)
        elif role == "label":
            labels.append(value)
        elif role == "mask":
            masks.append(value)

    return DatasetRoleInventory(
        format_name=format_name,
        image_paths=tuple(sorted(images)),
        label_paths=tuple(sorted(labels)),
        mask_paths=tuple(sorted(masks)),
    )


def require_manifest_inventory(
    manifest: dict[str, object],
    *,
    format_name: str,
) -> DatasetRoleInventory:
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        raise RuntimeError("dataset manifest files must be an array")
    relative_paths: list[str] = []
    for item in raw_files:
        if not isinstance(item, dict):
            raise RuntimeError("dataset manifest file entries must be objects")
        relative_path = item.get("relative_path")
        if not isinstance(relative_path, str) or not relative_path:
            raise RuntimeError("dataset manifest relative_path must be a non-empty string")
        relative_paths.append(relative_path)

    file_count = manifest.get("file_count")
    if type(file_count) is not int or file_count != len(relative_paths):
        raise RuntimeError("dataset manifest file_count does not match files")

    inventory = build_role_inventory(relative_paths, format_name=format_name)
    for field, expected in inventory.counts().items():
        actual = manifest.get(field)
        if type(actual) is not int or actual != expected:
            raise RuntimeError(f"dataset manifest {field} does not match canonical role inventory")
    return inventory


def _member_role(relative_path: str, *, format_name: str) -> str | None:
    path = PurePosixPath(relative_path)
    parts = path.parts
    suffix = path.suffix
    if format_name == "yolo":
        if len(parts) >= 2 and parts[0] == "images" and is_dataset_image_path(relative_path):
            return "image"
        if len(parts) >= 2 and parts[0] == "labels" and suffix == ".txt":
            return "label"
        return None
    if format_name == "coco":
        if len(parts) == 2 and parts[0] == "annotations" and suffix == ".json":
            return "label"
        if parts[0] != "annotations" and is_dataset_image_path(relative_path):
            return "image"
        return None
    if format_name == "imagenet":
        if (
            len(parts) >= 3
            and parts[0] in {"train", "val", "test"}
            and is_dataset_image_path(relative_path)
        ):
            return "image"
        return None
    if format_name == "anomalib":
        if _is_anomalib_source_role(parts) and suffix in _ANOMALIB_IMAGE_SUFFIXES:
            return "image"
        if _is_anomalib_mask_role(parts) and suffix == ".png":
            return "mask"
        return None
    raise AssertionError(f"unreachable dataset inventory format: {format_name}")


def _is_anomalib_source_role(parts: tuple[str, ...]) -> bool:
    return len(parts) == 3 and (parts[0], parts[1]) in _ANOMALIB_SOURCE_ROLES


def _is_anomalib_mask_role(parts: tuple[str, ...]) -> bool:
    return len(parts) == 4 and parts[:3] in {
        ("ground_truth", "val", "bad"),
        ("ground_truth", "test", "bad"),
    }
