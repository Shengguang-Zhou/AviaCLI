from __future__ import annotations

import concurrent.futures
import json
import sys
import time
from pathlib import Path
from typing import Any

from avia_sdk import errors as _errors
from avia_sdk.uploads.api import (
    _batch_upload_urls,
    _complete_dataset_file_batch,
    _complete_import,
    _create_dataset_session,
    _ensure_sha256_batch,
    _poll_import,
    _post_json,
    _put_file_with_retries,
)
from avia_sdk.uploads.manifest import (
    _guess_content_type,
    scan_source_manifest,
)
from avia_sdk.uploads.state import (
    _chunked,
    _load_resume_state,
    _save_state,
    _state_dir,
    _state_path,
)
from avia_sdk.uploads.timing import UploadTimingRecorder
from avia_sdk.uploads.urls import (
    upload_request_from_api as _upload_request_from_api,
)

_AviaHTTPError = _errors._AviaHTTPError
_UploadHTTPError = _errors._UploadHTTPError
_SUCCESS_STATUSES = {"succeeded", "success", "completed", "complete", "ready", "done"}
_FAILED_STATUSES = {"failed", "error", "cancelled", "canceled"}
_SUPPORTED_FORMATS = ("yolo", "coco", "imagenet")
_UPLOAD_CHUNK_SIZE = 4 * 1024 * 1024
_MAX_FOLDER_BATCH_SIZE = 1000
_DEFAULT_UPLOAD_READ_TIMEOUT = 45.0
_DEFAULT_UPLOAD_RETRY_BASE_DELAY = 0.25
_IMPORT_POLL_FAST_DELAYS_SEC = (0.25, 0.5, 1.0, 2.0, 4.0)

__all__ = (
    "_AviaHTTPError",
    "_UploadHTTPError",
    "create_source_import",
    "scan_source_manifest",
    "upload_dataset",
)



def create_source_import(
    *, api: str, token: object, project_id: str, payload: dict[str, object]
) -> dict[str, Any]:
    return _post_json(
        api=api,
        token=token,
        project_id=project_id,
        payload=payload,
    )


def upload_dataset(args: object, *, api: str, token: object) -> dict[str, Any]:
    upload_started = time.monotonic()
    upload_timing = UploadTimingRecorder(first_file_target=512)
    project_id = str(args.project)
    batch_size = int(args.batch_size)
    if batch_size < 1 or batch_size > _MAX_FOLDER_BATCH_SIZE:
        raise SystemExit(f"--batch-size must be between 1 and {_MAX_FOLDER_BATCH_SIZE}")
    max_files = getattr(args, "max_files", None)
    max_samples = getattr(args, "max_samples", None)
    manifest = scan_source_manifest(
        args.source,
        include_sha256=False,
        include_dimensions=False,
        hash_workers=int(args.hash_workers),
        max_files=max_files,
        format_name=str(args.format),
        max_samples=max_samples,
    )
    source_root = Path(str(manifest["source"]))
    files = list(manifest["files"])  # type: ignore[arg-type]
    state_dir = _state_dir(args)

    state = None
    if bool(args.resume):
        state = _load_resume_state(
            state_dir=state_dir,
            project_id=project_id,
            source=str(source_root),
            import_format=str(args.format),
        )

    if state is None:
        session = _create_dataset_session(
            api=api,
            token=token,
            project_id=project_id,
            manifest=manifest,
            args=args,
        )
        import_id = str(session.get("import_id") or "").strip()
        if not import_id:
            raise SystemExit("dataset-session response did not include import_id")
        state = {
            "api": api,
            "project_id": project_id,
            "import_id": import_id,
            "source": str(source_root),
            "format": str(args.format),
            "task_key": str(args.task_key),
            "files": {
                str(item["relative_path"]): {
                    "uploaded": False,
                    "size_bytes": int(item.get("size_bytes") or 0),
                    "sha256": str(item.get("sha256") or ""),
                    "width": int(item.get("width") or 0),
                    "height": int(item.get("height") or 0),
                }
                for item in files
            },
        }
        _save_state(state_dir, state)

    import_id = str(state["import_id"])
    state_files: dict[str, dict[str, Any]] = {
        str(key): dict(value) for key, value in dict(state.get("files") or {}).items()
    }
    file_by_relative = {str(item["relative_path"]): item for item in files}
    stream_flush_size = max(1, int(args.stream_flush_size))
    completion_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=max(1, int(args.batch_complete_concurrency))
    )
    completion_futures: list[
        tuple[concurrent.futures.Future[dict[str, Any]], list[str]]
    ] = []

    def stream_files_for(relative_paths: list[str]) -> list[dict[str, object]]:
        stream_files: list[dict[str, object]] = []
        for relative_path in relative_paths:
            item = dict(file_by_relative.get(relative_path) or {})
            if not item:
                item = {"relative_path": relative_path}
            item["relative_path"] = relative_path
            state_item = dict(state_files.get(relative_path) or {})
            if state_item.get("sha256") and not item.get("sha256"):
                item["sha256"] = state_item.get("sha256")
            if state_item.get("size_bytes") and not item.get("size_bytes"):
                item["size_bytes"] = state_item.get("size_bytes")
            if state_item.get("width") and not item.get("width"):
                item["width"] = state_item.get("width")
            if state_item.get("height") and not item.get("height"):
                item["height"] = state_item.get("height")
            if not item.get("content_type"):
                item["content_type"] = _guess_content_type(relative_path)
            stream_files.append(item)
        return stream_files

    def submit_stream_batch(relative_paths: list[str]) -> None:
        paths = list(dict.fromkeys(str(item) for item in relative_paths if str(item)))
        if not paths:
            return
        stream_files = stream_files_for(paths)
        future = completion_executor.submit(
            upload_timing.time_call,
            "batch_complete",
            _complete_dataset_file_batch,
            file_count=len(stream_files),
            byte_count=sum(int(item.get("size_bytes") or 0) for item in stream_files),
            api=api,
            token=token,
            project_id=project_id,
            import_id=import_id,
            files=stream_files,
            timeout=float(args.batch_complete_timeout),
            retries=int(args.batch_complete_retries),
        )
        completion_futures.append((future, paths))

    def record_streamed_batch(
        future: concurrent.futures.Future[dict[str, Any]], paths: list[str]
    ) -> None:
        batch_complete = future.result()
        for relative_path in paths:
            existing = state_files.get(relative_path, {})
            existing["streamed"] = True
            state_files[relative_path] = existing
        upload_timing.record_accepted_files(
            len(paths),
            elapsed_sec=time.monotonic() - upload_started,
        )
        state["files"] = state_files
        state["last_streaming_batch_complete"] = batch_complete
        _save_state(state_dir, state)

    def drain_stream_batches(*, block: bool = False) -> None:
        if not completion_futures:
            return
        if block:
            for future in concurrent.futures.as_completed(
                [future for future, _paths in completion_futures]
            ):
                pair = next(item for item in completion_futures if item[0] is future)
                completion_futures.remove(pair)
                record_streamed_batch(*pair)
            return
        for pair in list(completion_futures):
            future, _paths = pair
            if future.done():
                completion_futures.remove(pair)
                record_streamed_batch(*pair)

    def assert_all_files_streamed() -> None:
        missing_uploaded: list[str] = []
        missing_streamed: list[str] = []
        for item in files:
            relative_path = str(item["relative_path"])
            state_item = dict(state_files.get(relative_path) or {})
            if not bool(state_item.get("uploaded")):
                missing_uploaded.append(relative_path)
            elif not bool(state_item.get("streamed")):
                missing_streamed.append(relative_path)
        if not missing_uploaded and not missing_streamed:
            return
        raise SystemExit(
            "dataset upload did not finish streaming all files; "
            f"expected={len(files)} uploaded_missing={len(missing_uploaded)} "
            f"streamed_missing={len(missing_streamed)} "
            f"state_path={_state_path(state_dir, project_id, import_id)} "
            f"sample_uploaded_missing={missing_uploaded[:5]} "
            f"sample_streamed_missing={missing_streamed[:5]}"
        )

    try:
        for batch in _chunked(files, batch_size):
            drain_stream_batches()
            pending_upload = [
                item
                for item in batch
                if not bool(
                    state_files.get(str(item["relative_path"]), {}).get("uploaded")
                )
            ]
            pending_streamed_paths = [
                str(item["relative_path"])
                for item in batch
                if bool(state_files.get(str(item["relative_path"]), {}).get("uploaded"))
                and not bool(
                    state_files.get(str(item["relative_path"]), {}).get("streamed")
                )
            ]
            upload_items: dict[str, dict[str, Any]] = {}
            pending: list[dict[str, Any]] = []
            if pending_upload:
                pending = _ensure_sha256_batch(
                    source_root=source_root,
                    files=pending_upload,
                    hash_workers=int(args.hash_workers),
                )
                for item in pending:
                    file_by_relative[str(item["relative_path"])] = item
                urls = upload_timing.time_call(
                    "batch_upload_urls",
                    _batch_upload_urls,
                    file_count=len(pending),
                    byte_count=sum(
                        int(item.get("size_bytes") or 0) for item in pending
                    ),
                    api=api,
                    token=token,
                    project_id=project_id,
                    import_id=import_id,
                    files=pending,
                    timeout=float(args.batch_upload_url_timeout),
                    retries=int(args.batch_upload_url_retries),
                )
                upload_items = {
                    str(item.get("relative_path") or ""): dict(item)
                    for item in list(urls.get("files") or [])
                    if isinstance(item, dict)
                }
            elif not pending_streamed_paths:
                continue
            if pending_streamed_paths:
                submit_stream_batch(pending_streamed_paths)
            progress = {
                "started_at": time.monotonic(),
                "total_bytes": sum(
                    int(item.get("size_bytes") or 0) for item in pending
                ),
                "done_bytes": 0,
                "done_files": 0,
                "total_files": len(pending),
                "last_at": 0.0,
            }

            def emit_progress(
                *, force: bool = False, ctx: dict[str, float | int] = progress
            ) -> None:
                interval = max(0.0, float(args.progress_interval or 0.0))
                now = time.monotonic()
                if not force and (
                    interval <= 0 or now - float(ctx["last_at"]) < interval
                ):
                    return
                ctx["last_at"] = now
                elapsed = max(0.001, now - float(ctx["started_at"]))
                mib_done = int(ctx["done_bytes"]) / 1024 / 1024
                mib_total = int(ctx["total_bytes"]) / 1024 / 1024
                mibps = mib_done / elapsed
                print(
                    (
                        "folder upload progress: "
                        f"{int(ctx['done_files'])}/{int(ctx['total_files'])} files, "
                        f"{mib_done:.1f}/{mib_total:.1f} MiB, "
                        f"{mibps:.2f} MiB/s"
                    ),
                    file=sys.stderr,
                    flush=True,
                )

            def upload_one(
                relative_path: str,
                *,
                signed_items: dict[str, dict[str, Any]] = upload_items,
            ) -> tuple[str, dict[str, Any]]:
                signed = signed_items[relative_path]
                upload_url = str(
                    signed.get("upload_url") or signed.get("url") or ""
                ).strip()
                upload_url, upload_headers = _upload_request_from_api(
                    api,
                    upload_url,
                    upload_url_origin_override=str(
                        getattr(args, "upload_url_origin_override", "") or ""
                    ),
                )
                if not upload_url:
                    raise RuntimeError(f"upload URL missing for {relative_path}")
                source_file = source_root / relative_path
                headers = dict(signed.get("required_headers") or {})
                headers.update(upload_headers)
                upload_timing.time_call(
                    "file_put",
                    _put_file_with_retries,
                    file_count=1,
                    byte_count=source_file.stat().st_size,
                    upload_url=upload_url,
                    path=source_file,
                    headers=headers,
                    retries=int(args.upload_retries),
                    base_delay_sec=float(args.upload_retry_base_delay),
                    connect_timeout=float(args.upload_connect_timeout),
                    read_timeout=float(args.upload_read_timeout),
                )
                return relative_path, signed

            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=max(1, int(args.concurrency))
            )
            futures = [
                executor.submit(upload_one, str(item["relative_path"]))
                for item in pending
                if str(item["relative_path"]) in upload_items
            ]
            ready_to_stream: list[str] = []
            pending_state_saves = 0
            last_state_saved_at = time.monotonic()
            state_flush_every = max(1, int(args.state_flush_every))
            state_flush_interval = max(0.1, float(args.state_flush_interval))
            first_error: BaseException | None = None
            try:
                for future in concurrent.futures.as_completed(futures):
                    try:
                        relative_path, signed = future.result()
                    except BaseException as exc:
                        first_error = exc
                        for pending_future in futures:
                            pending_future.cancel()
                        break
                    existing = state_files.get(relative_path, {})
                    existing.update(
                        {
                            "uploaded": True,
                            "object_key": signed.get("object_key"),
                            "sha256": file_by_relative[relative_path].get("sha256"),
                            "size_bytes": file_by_relative[relative_path].get(
                                "size_bytes"
                            ),
                            "width": file_by_relative[relative_path].get("width"),
                            "height": file_by_relative[relative_path].get("height"),
                        }
                    )
                    state_files[relative_path] = existing
                    ready_to_stream.append(relative_path)
                    progress["done_files"] = int(progress["done_files"]) + 1
                    progress["done_bytes"] = int(progress["done_bytes"]) + int(
                        file_by_relative[relative_path].get("size_bytes") or 0
                    )
                    state["files"] = state_files
                    pending_state_saves += 1
                    now = time.monotonic()
                    if (
                        pending_state_saves >= state_flush_every
                        or now - last_state_saved_at >= state_flush_interval
                    ):
                        _save_state(state_dir, state)
                        pending_state_saves = 0
                        last_state_saved_at = now
                    emit_progress()
                    if len(ready_to_stream) >= stream_flush_size:
                        submit_stream_batch(ready_to_stream)
                        ready_to_stream = []
                    drain_stream_batches()
            finally:
                if pending_state_saves:
                    _save_state(state_dir, state)
                executor.shutdown(wait=first_error is None, cancel_futures=True)
            if first_error is not None:
                raise first_error
            emit_progress(force=True)
            if ready_to_stream:
                submit_stream_batch(ready_to_stream)
                drain_stream_batches()

            if pending:
                missing = [
                    str(item["relative_path"])
                    for item in pending
                    if not bool(
                        state_files.get(str(item["relative_path"]), {}).get("uploaded")
                    )
                ]
                if missing:
                    raise SystemExit(
                        f"server did not return upload URLs for: {', '.join(missing[:5])}"
                    )
        drain_stream_batches(block=True)
    finally:
        completion_executor.shutdown(wait=True, cancel_futures=False)
    assert_all_files_streamed()

    complete = upload_timing.time_call(
        "import_finalize",
        _complete_import,
        api=api,
        token=token,
        project_id=project_id,
        import_id=import_id,
    )
    state["completed"] = True
    state["complete_response"] = complete
    _save_state(state_dir, state)

    result: dict[str, Any] = {
        "import_id": import_id,
        "project_id": project_id,
        "file_count": int(manifest["file_count"]),
        "total_bytes": int(manifest["total_bytes"]),
        "complete": complete,
        "upload_timing": upload_timing.summary(),
        "state_path": str(_state_path(state_dir, project_id, import_id)),
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
