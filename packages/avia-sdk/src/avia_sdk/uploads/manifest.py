from __future__ import annotations

import concurrent.futures
import hashlib
import mimetypes
from pathlib import Path

from avia_sdk.uploads.metadata import read_yolo_class_names


def _guess_content_type(path: str) -> str:
    content_type, _encoding = mimetypes.guess_type(path)
    return content_type or "application/octet-stream"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_image_path(path: str) -> bool:
    return Path(str(path or "")).suffix.lower() in {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }


def _image_size_file(path: Path) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return int(image.width), int(image.height)
    except Exception as exc:
        raise RuntimeError(f"cannot read image dimensions for {path}: {exc}") from exc


def _manifest_item(
    *,
    source_root: Path,
    path: Path,
    include_sha256: bool,
    include_dimensions: bool = True,
) -> dict[str, object]:
    relative_path = path.relative_to(source_root).as_posix()
    item: dict[str, object] = {
        "relative_path": relative_path,
        "size_bytes": int(path.stat().st_size),
        "content_type": _guess_content_type(relative_path),
    }
    if _is_image_path(relative_path):
        item["width"], item["height"] = _image_size_file(path) if include_dimensions else (0, 0)
    if include_sha256:
        item["sha256"] = _sha256_file(path)
    return item


def scan_source_manifest(
    root: str | Path,
    *,
    include_sha256: bool = False,
    include_dimensions: bool = True,
    hash_workers: int = 1,
    max_files: int | None = None,
    format_name: str = "",
    max_samples: int | None = None,
) -> dict[str, object]:
    source_root = Path(root).expanduser().resolve()
    if not source_root.exists() or not source_root.is_dir():
        raise SystemExit(f"source path is not a directory: {source_root}")

    paths = []
    for path in sorted(item for item in source_root.rglob("*") if item.is_file()):
        if ".avia" in path.relative_to(source_root).parts:
            continue
        paths.append(path)
    if max_samples is not None:
        if str(format_name or "").strip().lower() != "yolo":
            raise SystemExit("--max-samples is only supported for YOLO folder uploads")
        paths = _limit_yolo_paths_by_samples(source_root, paths, int(max_samples))
    elif max_files is not None:
        if str(format_name or "").strip().lower() == "yolo":
            raise SystemExit("YOLO folder uploads must use --max-samples instead of --max-files")
        limit = int(max_files)
        if limit <= 0:
            raise SystemExit("--max-files must be greater than 0 when provided")
        paths = paths[:limit]

    def build_item(path: Path) -> dict[str, object]:
        return _manifest_item(
            source_root=source_root,
            path=path,
            include_sha256=include_sha256,
            include_dimensions=include_dimensions,
        )

    workers = max(1, int(hash_workers or 1))
    if workers == 1:
        files = [build_item(path) for path in paths]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            files = list(executor.map(build_item, paths))

    manifest: dict[str, object] = {
        "source": str(source_root),
        "file_count": len(files),
        "total_bytes": sum(int(item.get("size_bytes") or 0) for item in files),
        "files": files,
    }
    if str(format_name or "").strip().lower() == "yolo":
        manifest["classes"] = read_yolo_class_names(source_root)
    return manifest


def _limit_yolo_paths_by_samples(
    source_root: Path, paths: list[Path], max_samples: int
) -> list[Path]:
    if max_samples <= 0:
        raise SystemExit("--max-samples must be greater than 0 when provided")
    by_rel = {path.relative_to(source_root).as_posix(): path for path in paths}
    image_rels = [
        rel
        for rel in sorted(by_rel)
        if rel.startswith("images/")
        and Path(rel).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]
    selected_images = image_rels[:max_samples]
    selected: set[str] = set()
    for rel in sorted(by_rel):
        if rel in {
            "data.yaml",
            "data.yml",
            "dataset.yaml",
            "dataset.yml",
            "classes.txt",
        } or (rel.endswith(".json") and "/" not in rel):
            selected.add(rel)
    for rel in selected_images:
        selected.add(rel)
        stem = Path(rel).with_suffix("").as_posix()
        if stem.startswith("images/"):
            label_stem = "labels/" + stem[len("images/") :]
            label = f"{label_stem}.txt"
            if label in by_rel:
                selected.add(label)
    return [by_rel[rel] for rel in sorted(selected)]
