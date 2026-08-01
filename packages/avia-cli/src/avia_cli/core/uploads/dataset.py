from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from avia_cli.core.api_base import canonical_api_base
from avia_cli.core.errors import ConcurrentUploadError
from avia_cli.core.uploads.autotune import (
    compute_cpu_upload_params,
    compute_transport_concurrency,
    detect_storage_kind,
    probe_rtt_seconds,
)
from avia_cli.core.uploads.contracts import (
    require_folder_class_catalog,
    require_format_task,
)
from avia_cli.core.uploads.api import (
    _batch_upload_urls,
    _complete_dataset_file_batch,
    _complete_import,
    _create_dataset_session,
    _ensure_sha256_batch,
    _poll_import,
    _post_json,
    _put_file_with_retries,
)
from avia_cli.core.uploads.manifest import scan_source_manifest
from avia_cli.core.uploads.response_contracts import (
    IMPORT_TERMINAL_STATUSES,
    decode_source_import_response,
    validate_source_import_request,
)
from avia_cli.core.uploads.state import (
    _chunked,
    _exclusive_upload_state_lock,
    _load_resume_state,
    _save_state,
    _state_dir,
    _state_path,
)
from avia_cli.core.uploads.source_file import (
    SourceFileChangedError,
    SourceIdentity,
    assert_source_identity,
    capture_source_identity,
    open_verified_source,
)
from avia_cli.core.uploads.timing import UploadTimingRecorder
from avia_cli.core.uploads.transfer import (
    UploadTransportRoute,
    resolve_upload_route,
    validate_upload_contract,
)
from avia_cli.core.uploads.validation import require_valid_dataset

_MAX_FOLDER_BATCH_SIZE = 1000


@dataclass(frozen=True, slots=True)
class PreparedDatasetUpload:
    manifest: dict[str, object]
    source_root: Path
    files: list[dict[str, Any]]
    source_identities: dict[str, SourceIdentity]
    state_dir: Path
    validation_warnings: list[dict[str, Any]]


def create_source_import(
    *, api: str, token: object, project_id: str, payload: dict[str, object]
) -> dict[str, Any]:
    validate_source_import_request(payload)
    response = _post_json(
        api=api,
        token=token,
        project_id=project_id,
        payload=payload,
    )
    return decode_source_import_response(
        response,
        project_id=project_id,
        request_payload=payload,
    )


_AUTO_UPLOAD_PARAM_NAMES = (
    "concurrency",
    "batch_size",
    "hash_workers",
    "batch_complete_concurrency",
    "stream_flush_size",
)


def _capture_manifest_identities(
    source_root: Path,
    files: list[dict[str, Any]],
) -> dict[str, SourceIdentity]:
    identities: dict[str, SourceIdentity] = {}
    for item in files:
        relative_path = str(item["relative_path"])
        identity = capture_source_identity(source_root / relative_path)
        manifest_size = item.get("size_bytes")
        if isinstance(manifest_size, bool) or not isinstance(manifest_size, int):
            raise SourceFileChangedError(
                f"source manifest size is invalid: {relative_path}: manifest_size={manifest_size!r}"
            )
        if identity["size_bytes"] != manifest_size:
            raise SourceFileChangedError(
                "source file changed while scanning dataset: "
                f"{relative_path}: manifest_size={manifest_size} "
                f"actual_size={identity['size_bytes']}"
            )
        identities[relative_path] = identity
    return identities


def _assert_manifest_identities(
    source_root: Path,
    identities: dict[str, SourceIdentity],
    *,
    context: str,
) -> None:
    for relative_path, identity in identities.items():
        assert_source_identity(source_root / relative_path, identity, context=context)


def _assert_resume_source_unchanged(
    *,
    source_root: Path,
    files: list[dict[str, Any]],
    source_identities: dict[str, SourceIdentity],
    state: dict[str, Any],
) -> None:
    raw_state_files = state.get("files")
    if not isinstance(raw_state_files, dict):
        raise SourceFileChangedError("resume state files must be an object")
    current_paths = {str(item["relative_path"]) for item in files}
    state_paths = {str(path) for path in raw_state_files}
    if state_paths != current_paths:
        added = sorted(current_paths - state_paths)
        removed = sorted(state_paths - current_paths)
        raise SourceFileChangedError(
            f"resume source file set changed: added={added[:5]} removed={removed[:5]}"
        )

    for relative_path in sorted(current_paths):
        raw_item = raw_state_files[relative_path]
        if not isinstance(raw_item, dict):
            raise SourceFileChangedError(
                f"resume state file entry must be an object: {relative_path}"
            )
        expected_identity = raw_item.get("source_identity")
        current_identity = source_identities[relative_path]
        if expected_identity != current_identity:
            raise SourceFileChangedError(
                "resume source identity changed: "
                f"{relative_path}: expected={expected_identity!r} "
                f"actual={dict(current_identity)}"
            )
        state_size = raw_item.get("size_bytes")
        if (
            isinstance(state_size, bool)
            or not isinstance(state_size, int)
            or state_size != current_identity["size_bytes"]
        ):
            raise SourceFileChangedError(
                "resume source size changed: "
                f"{relative_path}: state_size={state_size!r} "
                f"actual_size={current_identity['size_bytes']}"
            )
        uploaded = raw_item.get("uploaded")
        if not isinstance(uploaded, bool):
            raise SourceFileChangedError(f"resume uploaded flag must be boolean: {relative_path}")
        if not uploaded:
            continue
        expected_sha256 = raw_item.get("sha256")
        if (
            not isinstance(expected_sha256, str)
            or len(expected_sha256) != 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            raise SourceFileChangedError(f"resume uploaded file digest is invalid: {relative_path}")
        digest = hashlib.sha256()
        path = source_root / relative_path
        with open_verified_source(path, current_identity) as source:
            for chunk in iter(lambda: source.handle.read(1024 * 1024), b""):
                digest.update(chunk)
            source.assert_unchanged(context="resume source changed while hashing uploaded file")
        if digest.hexdigest() != expected_sha256:
            raise SourceFileChangedError(f"resume uploaded file digest mismatch: {relative_path}")


def _resolve_local_upload_params(args: object) -> None:
    cores = os.cpu_count() or 4
    params = compute_cpu_upload_params(cores=cores)
    filled_any = False
    for name in _AUTO_UPLOAD_PARAM_NAMES:
        if name == "concurrency":
            continue
        if getattr(args, name, None) is None:
            setattr(args, name, params[name])
            filled_any = True
    if filled_any:
        print(
            "auto local upload params: "
            f"cores={cores} batch_complete={params['batch_complete_concurrency']} "
            f"hash_workers={params['hash_workers']} batch_size={params['batch_size']} "
            f"stream_flush={params['stream_flush_size']}",
            file=sys.stderr,
            flush=True,
        )


def _resolve_transport_concurrency(args: object, *, route: UploadTransportRoute) -> None:
    if getattr(args, "concurrency", None) is not None:
        return
    split = urlsplit(route.upload_url)
    host = split.hostname
    if split.scheme not in {"http", "https"} or not host:
        raise RuntimeError("signed upload URL must expose an absolute storage host")
    port = split.port or (443 if split.scheme == "https" else 80)
    storage_kind = detect_storage_kind(host)
    probe_rtt = probe_rtt_seconds(route, storage_kind)
    cores = os.cpu_count() or 4
    concurrency = compute_transport_concurrency(
        cores=cores,
        storage_kind=storage_kind,
        probe_rtt_s=probe_rtt,
    )
    vars(args)["concurrency"] = concurrency
    print(
        "auto upload transport: "
        f"storage_host={host}:{port} storage={storage_kind} "
        f"rtt={probe_rtt * 1000:.1f}ms concurrency={concurrency}",
        file=sys.stderr,
        flush=True,
    )


def prepare_dataset_upload(args: object) -> PreparedDatasetUpload:
    """Complete every local scan and dataset validation before authentication."""

    format_name, task_key = require_format_task(
        format_name=str(args.format),
        task_key=str(args.task_key),
    )
    declared_classes = list(args.class_name or []) or None
    if declared_classes is not None and format_name != "yolo":
        raise SystemExit("--class is only valid with --format yolo")
    configured_hash_workers = getattr(args, "hash_workers", None)
    validation_hash_workers = (
        int(configured_hash_workers)
        if configured_hash_workers is not None
        else min(16, max(4, os.cpu_count() or 4))
    )
    manifest = scan_source_manifest(
        args.source,
        include_dimensions=False,
        hash_workers=validation_hash_workers,
        format_name=format_name,
    )
    source_root = Path(str(manifest["source"]))
    files = list(manifest["files"])  # type: ignore[arg-type]
    source_identities = _capture_manifest_identities(source_root, files)
    classes, validation_warnings = require_valid_dataset(
        source_root=source_root,
        manifest=manifest,
        format_name=format_name,
        task_key=task_key,
        declared_classes=declared_classes,
    )
    if validation_warnings:
        print(
            json.dumps(
                {
                    "event": "dataset_validation_warnings",
                    "warning_count": len(validation_warnings),
                    "warnings": validation_warnings,
                },
                ensure_ascii=False,
            ),
            file=sys.stderr,
            flush=True,
        )
    _assert_manifest_identities(
        source_root,
        source_identities,
        context="source file changed during dataset validation",
    )
    manifest["classes"] = classes
    return PreparedDatasetUpload(
        manifest=manifest,
        source_root=source_root,
        files=files,
        source_identities=source_identities,
        state_dir=_state_dir(args),
        validation_warnings=validation_warnings,
    )


def upload_prepared_dataset(
    args: object,
    *,
    api: str,
    token: object,
    prepared: PreparedDatasetUpload,
) -> dict[str, Any]:
    api = canonical_api_base(api)
    upload_started = time.monotonic()
    upload_timing = UploadTimingRecorder(first_file_target=512)
    project_id = str(args.project)
    _assert_manifest_identities(
        prepared.source_root,
        prepared.source_identities,
        context="source file changed after dataset validation",
    )
    with _exclusive_upload_state_lock(
        state_dir=prepared.state_dir,
        project_id=project_id,
        source=str(prepared.source_root),
        import_format=str(args.format),
        task_key=str(args.task_key),
    ):
        _resolve_local_upload_params(args)
        result = _upload_validated_dataset(
            args=args,
            api=api,
            token=token,
            upload_started=upload_started,
            upload_timing=upload_timing,
            project_id=project_id,
            manifest=prepared.manifest,
            source_root=prepared.source_root,
            files=prepared.files,
            source_identities=prepared.source_identities,
            state_dir=prepared.state_dir,
        )
        result["validation_warnings"] = prepared.validation_warnings
        return result


def _upload_validated_dataset(
    *,
    args: object,
    api: str,
    token: object,
    upload_started: float,
    upload_timing: UploadTimingRecorder,
    project_id: str,
    manifest: dict[str, object],
    source_root: Path,
    files: list[dict[str, Any]],
    source_identities: dict[str, SourceIdentity],
    state_dir: Path,
) -> dict[str, Any]:
    batch_size = int(args.batch_size)
    if batch_size < 1 or batch_size > _MAX_FOLDER_BATCH_SIZE:
        raise SystemExit(f"--batch-size must be between 1 and {_MAX_FOLDER_BATCH_SIZE}")

    state = None
    if bool(args.resume):
        state = _load_resume_state(
            state_dir=state_dir,
            project_id=project_id,
            api=api,
            source=str(source_root),
            import_format=str(args.format),
            task_key=str(args.task_key),
        )
        if state is not None:
            _assert_resume_source_unchanged(
                source_root=source_root,
                files=files,
                source_identities=source_identities,
                state=state,
            )
        else:
            raise SystemExit(
                "--resume requested but no exact state exists for "
                f"project={project_id} task={args.task_key} source={source_root} api={api}"
            )

    if state is None:
        idempotency_key = str(uuid.uuid4())
        session_payload = _dataset_session_payload(
            manifest=manifest,
            args=args,
            idempotency_key=idempotency_key,
        )
        state = {
            "schema_version": 5,
            "phase": "session_pending",
            "api": api,
            "project_id": project_id,
            "import_id": None,
            "idempotency_key": idempotency_key,
            "session_payload": session_payload,
            "session_response": None,
            "source": str(source_root),
            "format": str(args.format),
            "task_key": str(args.task_key),
            "complete_response": None,
            "files": {
                str(item["relative_path"]): {
                    "uploaded": False,
                    "streamed": False,
                    "size_bytes": int(item.get("size_bytes") or 0),
                    "sha256": str(item.get("sha256") or ""),
                    "width": int(item.get("width") or 0),
                    "height": int(item.get("height") or 0),
                    "content_type": None,
                    "object_key": None,
                    "source_identity": dict(source_identities[str(item["relative_path"])]),
                }
                for item in files
            },
        }
        _save_state(state_dir, state)
    else:
        if state.get("phase") == "completed":
            raise SystemExit(
                f"resume state is already completed: "
                f"{_state_path(state_dir, project_id, str(state['idempotency_key']))}"
            )
        expected_session_payload = _dataset_session_payload(
            manifest=manifest,
            args=args,
            idempotency_key=str(state["idempotency_key"]),
        )
        if state.get("session_payload") != expected_session_payload:
            raise SourceFileChangedError(
                "resume session payload does not match the revalidated dataset: "
                f"expected={expected_session_payload!r} actual={state.get('session_payload')!r}"
            )

    if not str(state.get("import_id") or "").strip():
        session = _create_dataset_session(
            api=api,
            token=token,
            project_id=project_id,
            payload=dict(state["session_payload"]),
        )
        import_id = str(session.get("import_id") or "").strip()
        if not import_id:
            raise SystemExit("dataset-session response did not include import_id")
        state["import_id"] = import_id
        state["session_response"] = session
        state["phase"] = "uploading"
        _save_state(state_dir, state)

    import_id = str(state["import_id"])
    idempotency_key = str(state["idempotency_key"])
    state_files: dict[str, dict[str, Any]] = {
        str(key): dict(value) for key, value in dict(state.get("files") or {}).items()
    }
    file_by_relative = {str(item["relative_path"]): item for item in files}
    stream_flush_size = int(args.stream_flush_size)
    completion_executor = concurrent.futures.ThreadPoolExecutor(
        max_workers=int(args.batch_complete_concurrency)
    )
    completion_futures: list[tuple[concurrent.futures.Future[dict[str, Any]], list[str]]] = []
    completion_errors: list[tuple[str, list[str], BaseException]] = []

    def stream_files_for(relative_paths: list[str]) -> list[dict[str, object]]:
        stream_files: list[dict[str, object]] = []
        for relative_path in relative_paths:
            state_item = dict(state_files.get(relative_path) or {})
            if not state_item:
                raise RuntimeError(f"resume state has no file identity for {relative_path}")
            item: dict[str, object] = {
                "relative_path": relative_path,
                "object_key": state_item["object_key"],
                "size_bytes": state_item["size_bytes"],
                "content_type": state_item["content_type"],
                "sha256": state_item["sha256"],
            }
            if state_item["width"]:
                item["width"] = state_item["width"]
                item["height"] = state_item["height"]
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
        future.result()
        for relative_path in paths:
            existing = state_files.get(relative_path, {})
            existing["streamed"] = True
            state_files[relative_path] = existing
        upload_timing.record_accepted_files(
            len(paths),
            elapsed_sec=time.monotonic() - upload_started,
        )
        state["files"] = state_files
        _save_state(state_dir, state)

    def drain_stream_batches(*, block: bool = False) -> None:
        if not completion_futures:
            return
        if block:
            by_future = {future: pair for pair in completion_futures for future in (pair[0],)}
            pairs = [by_future[future] for future in concurrent.futures.as_completed(by_future)]
        else:
            pairs = [pair for pair in completion_futures if pair[0].done()]
        for pair in pairs:
            completion_futures.remove(pair)
            try:
                record_streamed_batch(*pair)
            except Exception as exc:
                completion_errors.append(("batch-complete", list(pair[1]), exc))

    def raise_stream_completion_errors() -> None:
        if not completion_errors:
            return
        drain_stream_batches(block=True)
        failures = list(completion_errors)
        completion_errors.clear()
        raise ConcurrentUploadError(failures)

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
            f"state_path={_state_path(state_dir, project_id, idempotency_key)} "
            f"sample_uploaded_missing={missing_uploaded[:5]} "
            f"sample_streamed_missing={missing_streamed[:5]}"
        )

    primary_error: Exception | None = None
    try:
        for batch in _chunked(files, batch_size):
            drain_stream_batches()
            raise_stream_completion_errors()
            pending_upload = [
                item
                for item in batch
                if not bool(state_files.get(str(item["relative_path"]), {}).get("uploaded"))
            ]
            pending_streamed_paths = [
                str(item["relative_path"])
                for item in batch
                if bool(state_files.get(str(item["relative_path"]), {}).get("uploaded"))
                and not bool(state_files.get(str(item["relative_path"]), {}).get("streamed"))
            ]
            upload_items: dict[str, dict[str, Any]] = {}
            upload_routes: dict[str, UploadTransportRoute] = {}
            pending: list[dict[str, Any]] = []
            if pending_upload:
                pending = _ensure_sha256_batch(
                    source_root=source_root,
                    files=pending_upload,
                    hash_workers=int(args.hash_workers),
                    source_identities=source_identities,
                )
                for item in pending:
                    file_by_relative[str(item["relative_path"])] = item
                signing_files = [
                    {
                        "relative_path": str(item["relative_path"]),
                        "size_bytes": int(item["size_bytes"]),
                        "sha256": str(item["sha256"]),
                    }
                    for item in pending
                ]
                urls = upload_timing.time_call(
                    "batch_upload_urls",
                    _batch_upload_urls,
                    file_count=len(pending),
                    byte_count=sum(int(item.get("size_bytes") or 0) for item in pending),
                    api=api,
                    token=token,
                    project_id=project_id,
                    import_id=import_id,
                    files=signing_files,
                    timeout=float(args.batch_upload_url_timeout),
                    retries=int(args.batch_upload_url_retries),
                )
                upload_items = {str(item["relative_path"]): dict(item) for item in urls["files"]}
                for relative_path, signed_item in upload_items.items():
                    validate_upload_contract(
                        upload_url=str(signed_item["upload_url"]),
                        headers=dict(signed_item["required_headers"]),
                        expected_length=source_identities[relative_path]["size_bytes"],
                    )
                    upload_routes[relative_path] = resolve_upload_route(
                        str(signed_item["upload_url"])
                    )
                first_relative_path = str(urls["files"][0]["relative_path"])
                _resolve_transport_concurrency(args, route=upload_routes[first_relative_path])
            elif not pending_streamed_paths:
                continue
            if pending_streamed_paths:
                submit_stream_batch(pending_streamed_paths)
            progress = {
                "started_at": time.monotonic(),
                "total_bytes": sum(int(item.get("size_bytes") or 0) for item in pending),
                "done_bytes": 0,
                "done_files": 0,
                "total_files": len(pending),
                "last_at": 0.0,
            }

            def emit_progress(
                *, force: bool = False, ctx: dict[str, float | int] = progress
            ) -> None:
                interval = float(args.progress_interval)
                now = time.monotonic()
                if not force and now - float(ctx["last_at"]) < interval:
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
                routes: dict[str, UploadTransportRoute] = upload_routes,
            ) -> tuple[str, dict[str, Any]]:
                signed = signed_items[relative_path]
                upload_url = str(signed["upload_url"])
                if not upload_url:
                    raise RuntimeError(f"upload URL missing for {relative_path}")
                source_file = source_root / relative_path
                headers = dict(signed.get("required_headers") or {})
                upload_timing.time_call(
                    "file_put",
                    _put_file_with_retries,
                    file_count=1,
                    byte_count=source_identities[relative_path]["size_bytes"],
                    route=routes[relative_path],
                    path=source_file,
                    expected_identity=source_identities[relative_path],
                    headers=headers,
                    retries=int(args.upload_retries),
                    base_delay_sec=float(args.upload_retry_base_delay),
                    connect_timeout=float(args.upload_connect_timeout),
                    read_timeout=float(args.upload_read_timeout),
                )
                return relative_path, signed

            executor = concurrent.futures.ThreadPoolExecutor(max_workers=int(args.concurrency or 1))
            futures = {
                executor.submit(upload_one, str(item["relative_path"])): str(item["relative_path"])
                for item in pending
                if str(item["relative_path"]) in upload_items
            }
            ready_to_stream: list[str] = []
            pending_state_saves = 0
            last_state_saved_at = time.monotonic()
            state_flush_every = int(args.state_flush_every)
            state_flush_interval = float(args.state_flush_interval)
            put_failures: list[tuple[str, list[str], BaseException]] = []
            try:
                for future in concurrent.futures.as_completed(futures):
                    if future.cancelled():
                        continue
                    try:
                        relative_path, signed = future.result()
                    except Exception as exc:
                        put_failures.append(("file-put", [futures[future]], exc))
                        if len(put_failures) == 1:
                            for pending_future in futures:
                                pending_future.cancel()
                        continue
                    existing = state_files.get(relative_path, {})
                    existing.update(
                        {
                            "uploaded": True,
                            "object_key": signed.get("object_key"),
                            "content_type": signed.get("content_type"),
                            "sha256": str(file_by_relative[relative_path].get("sha256") or ""),
                            "size_bytes": int(
                                file_by_relative[relative_path].get("size_bytes") or 0
                            ),
                            "width": int(file_by_relative[relative_path].get("width") or 0),
                            "height": int(file_by_relative[relative_path].get("height") or 0),
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
                # Running PUTs cannot be cancelled safely. Wait for them and
                # persist every completed success before returning the first
                # failure, so no background thread keeps mutating object
                # storage after the command has exited and resume state remains
                # an exact account of remote side effects.
                executor.shutdown(wait=True, cancel_futures=True)
            if put_failures:
                drain_stream_batches(block=True)
                failures = [*put_failures, *completion_errors]
                completion_errors.clear()
                raise ConcurrentUploadError(failures)
            emit_progress(force=True)
            if ready_to_stream:
                submit_stream_batch(ready_to_stream)
                drain_stream_batches()
            raise_stream_completion_errors()

            if pending:
                missing = [
                    str(item["relative_path"])
                    for item in pending
                    if not bool(state_files.get(str(item["relative_path"]), {}).get("uploaded"))
                ]
                if missing:
                    raise SystemExit(
                        f"server did not return upload URLs for: {', '.join(missing[:5])}"
                    )
        drain_stream_batches(block=True)
        raise_stream_completion_errors()
    except Exception as exc:
        primary_error = exc
    finally:
        drain_stream_batches(block=True)
        completion_executor.shutdown(wait=True, cancel_futures=False)
    if primary_error is not None:
        failures = (
            list(primary_error.failures)
            if isinstance(primary_error, ConcurrentUploadError)
            else [("folder-upload", [], primary_error)]
        )
        failures.extend(completion_errors)
        raise ConcurrentUploadError(failures) from primary_error
    raise_stream_completion_errors()
    assert_all_files_streamed()

    complete = upload_timing.time_call(
        "import_finalize",
        _complete_import,
        api=api,
        token=token,
        project_id=project_id,
        import_id=import_id,
        request_payload=dict(state["session_payload"]),
        session_response=dict(state["session_response"]),
    )
    state["phase"] = "completed"
    state["complete_response"] = complete
    _save_state(state_dir, state)

    result: dict[str, Any] = {
        "import_id": import_id,
        "project_id": project_id,
        "format": str(args.format),
        "task_key": str(args.task_key),
        "file_count": int(manifest["file_count"]),
        "total_bytes": int(manifest["total_bytes"]),
        "image_count": int(manifest["image_count"]),
        "label_count": int(manifest["label_count"]),
        "mask_count": int(manifest["mask_count"]),
        "complete": complete,
        "upload_timing": upload_timing.summary(),
        "state_path": str(_state_path(state_dir, project_id, idempotency_key)),
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
        if status not in IMPORT_TERMINAL_STATUSES:
            raise RuntimeError(f"poll returned non-terminal status: {status!r}")
        if status == "failed":
            raise SystemExit(json.dumps(result, ensure_ascii=False))
    result["dataset_manifest_ref"] = complete["dataset_manifest_ref"]
    result["read_lease"] = complete["read_lease"]
    return result


def _dataset_session_payload(
    *, manifest: dict[str, object], args: object, idempotency_key: str
) -> dict[str, object]:
    classes = require_folder_class_catalog(
        list(getattr(args, "class_name", None) or []) or manifest.get("classes"),
        format_name=str(args.format),
        label="dataset session classes",
    )
    return {
        "idempotency_key": idempotency_key,
        "format": str(args.format),
        "root_name": Path(str(manifest["source"])).name,
        "task_key": str(args.task_key),
        "classes": classes,
        "file_count": int(manifest["file_count"]),
        "total_bytes": int(manifest["total_bytes"]),
    }
