from __future__ import annotations

import concurrent.futures
import json
import unicodedata
from pathlib import Path

from avia_cli.core.uploads.inventory import (
    build_role_inventory,
    is_dataset_image_path,
    requires_lowercase_media_suffix,
)
from avia_cli.core.uploads.image_validation import (
    ImageEncodingMismatch,
    decoded_image_size,
)
from avia_cli.core.uploads.metadata import read_yolo_class_names


class ManifestImageError(RuntimeError):
    def __init__(
        self,
        *,
        relative_path: str,
        detail: str,
        expected_format: str | None = None,
        actual_format: str | None = None,
    ) -> None:
        self.relative_path = relative_path
        self.expected_format = expected_format
        self.actual_format = actual_format
        super().__init__(detail)


def is_client_state_path(relative_path: Path) -> bool:
    parts = relative_path.parts
    return len(parts) > 2 and parts[:2] == (".avia", "imports")


def is_client_state_directory(relative_path: Path) -> bool:
    return relative_path.parts in {(".avia",), (".avia", "imports")} or is_client_state_path(
        relative_path
    )


def _image_size_file(path: Path) -> tuple[int, int]:
    return decoded_image_size(path)


def _manifest_item(
    *,
    source_root: Path,
    path: Path,
    format_name: str,
    include_dimensions: bool = True,
) -> dict[str, object]:
    relative_path = _canonical_relative_path(
        path.relative_to(source_root),
        format_name=format_name,
    )
    item: dict[str, object] = {
        "relative_path": relative_path,
        "size_bytes": int(path.stat().st_size),
    }
    if is_dataset_image_path(relative_path):
        try:
            width, height = _image_size_file(path)
        except ImageEncodingMismatch as exc:
            raise ManifestImageError(
                relative_path=relative_path,
                detail=(
                    "image encoding does not match its declared suffix: "
                    f"path={relative_path} expected={exc.expected_format} "
                    f"actual={exc.actual_format}"
                ),
                expected_format=exc.expected_format,
                actual_format=exc.actual_format,
            ) from exc
        except ValueError as exc:
            raise ManifestImageError(
                relative_path=relative_path,
                detail=f"cannot read image dimensions: path={relative_path} error={exc}",
            ) from exc
        item["width"], item["height"] = (width, height) if include_dimensions else (0, 0)
    return item


def scan_source_manifest(
    root: str | Path,
    *,
    include_dimensions: bool = True,
    hash_workers: int = 1,
    format_name: str,
) -> dict[str, object]:
    requested_root = Path(root).expanduser()
    if requested_root.is_symlink():
        _raise_manifest_error(
            "dataset_symlink",
            "dataset source root must not be a symbolic link",
            path=str(requested_root),
        )
    if (
        requested_root.name != requested_root.name.strip()
        or requested_root.name != unicodedata.normalize("NFC", requested_root.name)
        or "\\" in requested_root.name
        or any(unicodedata.category(character) == "Cc" for character in requested_root.name)
    ):
        _raise_manifest_error(
            "invalid_dataset_path",
            "dataset source root name must not contain control characters",
            path=str(requested_root),
        )
    source_root = requested_root.resolve()
    if not source_root.exists() or not source_root.is_dir():
        raise SystemExit(f"source path is not a directory: {source_root}")

    paths: list[Path] = []
    for path in sorted(source_root.rglob("*")):
        relative = path.relative_to(source_root)
        relative_path = _canonical_relative_path(relative, format_name=format_name)
        if path.is_symlink():
            _raise_manifest_error(
                "dataset_symlink",
                "dataset source tree must not contain symbolic links",
                path=relative_path,
            )
        if any(unicodedata.category(character) == "Cc" for character in relative_path):
            _raise_manifest_error(
                "invalid_dataset_path",
                "dataset paths must not contain control characters",
                path=relative_path,
            )
        if path.is_dir():
            continue
        if not path.is_file():
            _raise_manifest_error(
                "unsupported_dataset_member",
                "dataset source tree must contain only regular files and directories",
                path=relative_path,
            )
        if is_client_state_path(relative):
            continue
        paths.append(path)

    def build_item(path: Path) -> dict[str, object]:
        return _manifest_item(
            source_root=source_root,
            path=path,
            format_name=format_name,
            include_dimensions=include_dimensions,
        )

    workers = int(hash_workers)
    if workers <= 0:
        raise ValueError("hash_workers must be greater than zero")
    if workers == 1:
        files = [build_item(path) for path in paths]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            files = list(executor.map(build_item, paths))

    inventory = build_role_inventory(
        (str(item["relative_path"]) for item in files),
        format_name=format_name,
    )
    manifest: dict[str, object] = {
        "source": str(source_root),
        "file_count": len(files),
        "total_bytes": sum(int(item.get("size_bytes") or 0) for item in files),
        "files": files,
        **inventory.counts(),
    }
    if format_name == "yolo":
        manifest["classes"] = read_yolo_class_names(source_root)
    return manifest


def _raise_manifest_error(code: str, message: str, *, path: str) -> None:
    raise SystemExit(
        json.dumps(
            {"code": code, "message": message, "path": path},
            ensure_ascii=False,
        )
    )


def _canonical_relative_path(relative: Path, *, format_name: str) -> str:
    raw = relative.as_posix()
    normalized = unicodedata.normalize("NFC", raw)
    suffix = relative.suffix
    invalid = (
        not raw
        or raw != normalized
        or "\\" in raw
        or raw != raw.strip()
        or raw.startswith("/")
        or any(part in {"", ".", ".."} or part != part.strip() for part in raw.split("/"))
        or any(unicodedata.category(character) == "Cc" for character in raw)
        or (
            requires_lowercase_media_suffix(raw, format_name=format_name)
            and suffix != suffix.lower()
        )
    )
    if invalid:
        _raise_manifest_error(
            "invalid_dataset_path",
            "dataset members must use unique canonical NFC POSIX relative paths",
            path=raw,
        )
    return raw
