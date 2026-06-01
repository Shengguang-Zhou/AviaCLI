from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Iterable
from urllib import parse

from avia_sdk.uploads.manifest import (
    _image_size_file,
    _is_image_path,
    _sha256_file,
)


def _source_import_payload(args: argparse.Namespace) -> dict[str, object]:
    return {
        "source_kind": str(args.source_kind),
        "uri": str(args.source),
        "format": str(args.format),
        "task_key": str(args.task_key),
        "classes": list(args.class_name or []),
        "auto_post_processing": bool(args.auto_post_processing),
    }


def _chunked(items: list[dict[str, object]], size: int) -> Iterable[list[dict[str, object]]]:
    step = max(1, int(size or 1))
    for idx in range(0, len(items), step):
        yield items[idx : idx + step]


def _item_with_upload_metadata(
    source_root: Path,
    item: dict[str, object],
    *,
    sha256_file: Callable[[Path], str] = _sha256_file,
    image_size_file: Callable[[Path], tuple[int, int]] = _image_size_file,
    is_image_path: Callable[[str], bool] = _is_image_path,
) -> dict[str, object]:
    enriched = dict(item)
    relative_path = str(item["relative_path"])
    path = source_root / relative_path
    if not str(enriched.get("sha256") or "").strip():
        enriched["sha256"] = sha256_file(path)
    if is_image_path(relative_path) and (
        not int(enriched.get("width") or 0) or not int(enriched.get("height") or 0)
    ):
        width, height = image_size_file(path)
        enriched["width"] = width
        enriched["height"] = height
    return enriched


def _ensure_sha256_batch(
    *,
    source_root: Path,
    files: list[dict[str, object]],
    hash_workers: int,
    sha256_file: Callable[[Path], str] = _sha256_file,
    image_size_file: Callable[[Path], tuple[int, int]] = _image_size_file,
    is_image_path: Callable[[str], bool] = _is_image_path,
) -> list[dict[str, object]]:
    missing = [
        item
        for item in files
        if not str(item.get("sha256") or "").strip()
        or (
            is_image_path(str(item["relative_path"]))
            and (not int(item.get("width") or 0) or not int(item.get("height") or 0))
        )
    ]
    if not missing:
        return files

    workers = max(1, int(hash_workers or 1))
    hashed_by_relative: dict[str, dict[str, object]] = {}
    if workers == 1 or len(missing) == 1:
        for item in missing:
            hashed = _item_with_upload_metadata(
                source_root,
                item,
                sha256_file=sha256_file,
                image_size_file=image_size_file,
                is_image_path=is_image_path,
            )
            hashed_by_relative[str(hashed["relative_path"])] = hashed
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            for hashed in executor.map(
                lambda item: _item_with_upload_metadata(
                    source_root,
                    item,
                    sha256_file=sha256_file,
                    image_size_file=image_size_file,
                    is_image_path=is_image_path,
                ),
                missing,
            ):
                hashed_by_relative[str(hashed["relative_path"])] = hashed

    return [hashed_by_relative.get(str(item["relative_path"]), item) for item in files]


def _safe_state_segment(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    return cleaned.strip("._") or "default"


def _state_dir(args: argparse.Namespace) -> Path:
    explicit = str(getattr(args, "state_dir", "") or os.environ.get("AVIA_STATE_DIR", "")).strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    return Path(str(args.source)).expanduser().resolve() / ".avia" / "imports"


def _state_path(state_dir: Path, project_id: str, import_id: str) -> Path:
    return state_dir / _safe_state_segment(project_id) / f"{_safe_state_segment(import_id)}.json"


def _save_state(state_dir: Path, state: dict[str, Any]) -> None:
    path = _state_path(
        state_dir,
        str(state.get("project_id") or "project"),
        str(state.get("import_id") or "import"),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp.replace(path)


def _load_resume_state(
    *,
    state_dir: Path,
    project_id: str,
    source: str,
    import_format: str,
) -> dict[str, Any] | None:
    project_dir = state_dir / _safe_state_segment(project_id)
    if not project_dir.exists():
        return None
    candidates = sorted(
        project_dir.glob("*.json"), key=lambda item: item.stat().st_mtime, reverse=True
    )
    for path in candidates:
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if (
            str(state.get("source") or "") == str(source)
            and str(state.get("format") or "") == str(import_format)
            and str(state.get("project_id") or "") == str(project_id)
            and str(state.get("import_id") or "")
        ):
            return state
    return None


def _should_bypass_proxy_for_upload(parsed: parse.SplitResult) -> bool:
    host = str(parsed.hostname or "").lower()
    if not host:
        return False
    bypass_hosts = {
        entry.strip().lower()
        for entry in os.environ.get("AVIA_UPLOAD_NO_PROXY_HOSTS", "").split(",")
        if entry.strip()
    }
    if not bypass_hosts:
        bypass_hosts = {"localhost", "127.0.0.1", "minio"}
    return any(host == item or host.endswith("." + item) for item in bypass_hosts)
