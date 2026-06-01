from __future__ import annotations

import argparse
import concurrent.futures
import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Any
from urllib import parse

from avia_sdk.http.form import _request_form_json
from avia_sdk.uploads.api import (
    _complete_import,
    _poll_import,
    _project_path,
    _request_json,
)
from avia_sdk.uploads.dataset import _FAILED_STATUSES
from avia_sdk.uploads.parts import _put_file_part_with_retries
from avia_sdk.uploads.state import _chunked
from avia_sdk.uploads.urls import upload_url_from_api

_AUTO_ARCHIVE_PART_SIZE_MB = 64
_MAX_ARCHIVE_MULTIPART_PARTS = 9000
_ARCHIVE_MULTIPART_SUFFIX = {
    "yolo": "imports/yolo-zip/multipart-upload",
    "coco": "imports/coco-zip/multipart-upload",
    "imagenet": "imports/imagenet-zip/multipart-upload",
}


def _default_archive_path(source_root: Path) -> Path:
    return source_root / ".avia" / "archives" / f"{source_root.name}.zip"


def _create_zip_archive(
    *,
    source_root: Path,
    archive_path: Path,
    force: bool,
) -> dict[str, Any]:
    if archive_path.exists() and not force:
        return {
            "archive_path": str(archive_path),
            "created": False,
            "size_bytes": int(archive_path.stat().st_size),
            "file_count": None,
        }
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = archive_path.with_suffix(archive_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    file_count = 0
    with zipfile.ZipFile(
        tmp,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as zf:
        for path in sorted(item for item in source_root.rglob("*") if item.is_file()):
            relative = path.relative_to(source_root)
            if ".avia" in relative.parts:
                continue
            zf.write(path, relative.as_posix())
            file_count += 1
    tmp.replace(archive_path)
    return {
        "archive_path": str(archive_path),
        "created": True,
        "size_bytes": int(archive_path.stat().st_size),
        "file_count": file_count,
    }


def _create_archive_multipart_upload(
    *,
    api: str,
    token: str,
    project_id: str,
    import_format: str,
    file_name: str,
    expires_in: int,
) -> dict[str, Any]:
    suffix = _ARCHIVE_MULTIPART_SUFFIX.get(str(import_format))
    if not suffix:
        raise SystemExit(f"archive multipart upload is not supported for format: {import_format}")
    return _request_form_json(
        method="POST",
        url=_project_path(api, project_id, suffix),
        token=token,
        fields={"file_name": file_name, "expires_in": int(expires_in)},
        timeout=60,
    )


def _sign_archive_multipart_parts(
    *,
    api: str,
    token: str,
    project_id: str,
    import_id: str,
    part_numbers: list[int],
    expires_in: int,
) -> dict[str, Any]:
    return _request_json(
        method="POST",
        url=(
            _project_path(
                api,
                project_id,
                f"imports/{parse.quote(import_id, safe='')}/multipart/parts:sign",
            )
            + f"?expires_in={int(expires_in)}"
        ),
        token=token,
        payload={"part_numbers": list(part_numbers)},
        timeout=60,
    )


def _complete_archive_multipart_upload(
    *,
    api: str,
    token: str,
    project_id: str,
    import_id: str,
    upload_id: str,
    parts: list[dict[str, object]],
) -> dict[str, Any]:
    return _request_json(
        method="POST",
        url=_project_path(
            api,
            project_id,
            f"imports/{parse.quote(import_id, safe='')}/multipart/complete",
        ),
        token=token,
        payload={"upload_id": upload_id, "parts": parts},
        timeout=120,
    )


def _archive_part_ranges(path: Path, part_size: int) -> list[dict[str, int]]:
    total = int(path.stat().st_size)
    ranges: list[dict[str, int]] = []
    part_number = 1
    for offset in range(0, total, int(part_size)):
        size = min(int(part_size), total - offset)
        ranges.append({"part_number": part_number, "offset": int(offset), "size_bytes": int(size)})
        part_number += 1
    if not ranges:
        raise SystemExit(f"archive is empty: {path}")
    return ranges


def _resolve_archive_part_size(*, total_bytes: int, requested_mb: int) -> int:
    requested = int(requested_mb or 0)
    if requested > 0:
        return max(5 * 1024 * 1024, requested * 1024 * 1024)
    min_for_part_limit = (int(total_bytes) + _MAX_ARCHIVE_MULTIPART_PARTS - 1) // (
        _MAX_ARCHIVE_MULTIPART_PARTS
    )
    return max(
        5 * 1024 * 1024,
        _AUTO_ARCHIVE_PART_SIZE_MB * 1024 * 1024,
        int(min_for_part_limit),
    )


def _upload_archive_multipart_parts(
    *,
    api: str,
    token: str,
    project_id: str,
    import_id: str,
    archive_path: Path,
    upload_id: str,
    part_size: int,
    part_sign_batch_size: int,
    concurrency: int,
    progress_interval_sec: float,
    expires_in: int,
    retries: int,
    base_delay_sec: float,
) -> list[dict[str, object]]:
    ranges = _archive_part_ranges(archive_path, part_size)
    uploaded: dict[int, dict[str, object]] = {}
    range_by_part = {int(item["part_number"]): item for item in ranges}
    total_bytes = sum(int(item["size_bytes"]) for item in ranges)
    completed_bytes = 0
    completed_parts = 0
    upload_started_at = time.monotonic()
    last_progress_at = 0.0

    def emit_progress(*, force: bool = False) -> None:
        nonlocal last_progress_at
        now = time.monotonic()
        interval = max(0.0, float(progress_interval_sec or 0.0))
        if not force and (interval <= 0 or now - last_progress_at < interval):
            return
        last_progress_at = now
        elapsed = max(0.001, now - upload_started_at)
        mib_done = completed_bytes / 1024 / 1024
        mib_total = total_bytes / 1024 / 1024
        mibps = mib_done / elapsed
        print(
            (
                "archive upload progress: "
                f"{completed_parts}/{len(ranges)} parts, "
                f"{mib_done:.1f}/{mib_total:.1f} MiB, "
                f"{mibps:.2f} MiB/s"
            ),
            file=sys.stderr,
            flush=True,
        )

    for batch in _chunked(
        [{"part_number": int(item["part_number"])} for item in ranges],
        max(1, int(part_sign_batch_size)),
    ):
        part_numbers = [int(item["part_number"]) for item in batch]
        signed = _sign_archive_multipart_parts(
            api=api,
            token=token,
            project_id=project_id,
            import_id=import_id,
            part_numbers=part_numbers,
            expires_in=expires_in,
        )
        signed_by_part = {
            int(item.get("part_number") or 0): dict(item)
            for item in list(signed.get("parts") or [])
            if isinstance(item, dict)
        }

        def upload_one(
            part_number: int,
            *,
            signed_items: dict[int, dict[str, Any]] = signed_by_part,
        ) -> dict[str, object]:
            signed_part = signed_items.get(part_number)
            if not signed_part:
                raise RuntimeError(f"signed multipart URL missing for part {part_number}")
            upload_url = str(signed_part.get("upload_url") or "").strip()
            upload_url = upload_url_from_api(api, upload_url)
            if not upload_url:
                raise RuntimeError(f"multipart upload URL missing for part {part_number}")
            part = range_by_part[part_number]
            etag = _put_file_part_with_retries(
                upload_url=upload_url,
                path=archive_path,
                offset=int(part["offset"]),
                size=int(part["size_bytes"]),
                headers=dict(signed_part.get("required_headers") or {}),
                retries=retries,
                base_delay_sec=base_delay_sec,
            )
            return {
                "part_number": part_number,
                "etag": etag,
                "size_bytes": int(part["size_bytes"]),
            }

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, int(concurrency))
        ) as executor:
            futures = [executor.submit(upload_one, part_number) for part_number in part_numbers]
            first_error: BaseException | None = None
            for future in concurrent.futures.as_completed(futures):
                try:
                    part_result = future.result()
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
                    continue
                uploaded[int(part_result["part_number"])] = part_result
                completed_parts += 1
                completed_bytes += int(part_result["size_bytes"])
                emit_progress()
            if first_error is not None:
                raise first_error

    emit_progress(force=True)
    return [uploaded[int(item["part_number"])] for item in ranges]


def dataset_upload_archive(args: argparse.Namespace, *, api: str, token: object) -> dict[str, Any]:
    project_id = str(args.project)
    source_root = Path(args.source).expanduser().resolve()
    if not source_root.exists() or not source_root.is_dir():
        raise SystemExit(f"source path is not a directory: {source_root}")
    archive_path = (
        Path(str(args.archive_path)).expanduser().resolve()
        if str(args.archive_path or "").strip()
        else _default_archive_path(source_root)
    )
    started = time.monotonic()
    archive = _create_zip_archive(
        source_root=source_root,
        archive_path=archive_path,
        force=bool(args.force_archive),
    )
    archive["elapsed_sec"] = round(time.monotonic() - started, 3)
    part_size = _resolve_archive_part_size(
        total_bytes=int(archive.get("size_bytes") or 0),
        requested_mb=int(args.multipart_part_size_mb),
    )
    upload_started = time.monotonic()
    transport = str(args.transport or "object-storage").strip().lower()
    multipart_complete: dict[str, Any] | None = None
    upload_id: str | None = None
    uploaded_parts: list[dict[str, object]] = []
    if transport == "object-storage":
        signed = _create_archive_multipart_upload(
            api=api,
            token=token,
            project_id=project_id,
            import_format=str(args.format),
            file_name=archive_path.name,
            expires_in=int(args.expires_in),
        )
        import_id = str(signed.get("import_id") or "").strip()
        if not import_id:
            raise SystemExit("archive upload-url response did not include import_id")
        upload_id = str(signed.get("upload_id") or "").strip()
        if not upload_id:
            raise SystemExit("archive multipart response did not include upload_id")
        uploaded_parts = _upload_archive_multipart_parts(
            api=api,
            token=token,
            project_id=project_id,
            import_id=import_id,
            archive_path=archive_path,
            upload_id=upload_id,
            part_size=part_size,
            part_sign_batch_size=int(args.multipart_sign_batch_size),
            concurrency=int(args.multipart_concurrency),
            progress_interval_sec=float(args.progress_interval),
            expires_in=int(args.expires_in),
            retries=int(args.upload_retries),
            base_delay_sec=float(args.upload_retry_base_delay),
        )
        multipart_complete = _complete_archive_multipart_upload(
            api=api,
            token=token,
            project_id=project_id,
            import_id=import_id,
            upload_id=upload_id,
            parts=uploaded_parts,
        )
    else:
        raise SystemExit(f"unsupported archive upload transport: {transport}")
    complete = _complete_import(
        api=api, token=token, project_id=project_id, import_id=import_id
    )
    result: dict[str, Any] = {
        "import_id": import_id,
        "project_id": project_id,
        "archive": archive,
        "transport": transport,
        "upload_id": upload_id,
        "multipart": {
            "part_size_bytes": part_size,
            "part_count": len(uploaded_parts),
            "complete": multipart_complete,
        },
        "archive_upload_elapsed_sec": round(time.monotonic() - upload_started, 3),
        "total_bytes": int(archive.get("size_bytes") or 0),
        "complete": complete,
    }
    if bool(args.wait):
        poll = _poll_import(
            api=api,
            token=token,
            project_id=project_id,
            import_id=import_id,
            timeout_sec=int(args.wait_timeout),
            interval_sec=float(args.poll_interval),
        )
        result["job"] = poll
        status = str(poll.get("status") or "").strip().lower()
        if status in _FAILED_STATUSES:
            raise SystemExit(json.dumps(result, ensure_ascii=False))
    return result
