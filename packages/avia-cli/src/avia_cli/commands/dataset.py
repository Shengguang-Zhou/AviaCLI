from __future__ import annotations

import json

from avia_cli.context import api_from_args, token_from_args
from avia_cli.core.uploads.archive import dataset_upload_archive
from avia_cli.core.uploads.dataset import upload_dataset
from avia_cli.core.uploads.inspect import (
    build_cleanup_plan,
    inspect_dataset,
    verify_dataset,
)
from avia_cli.core.uploads.manifest import scan_source_manifest


def handle_dataset_command(args) -> int:
    if args.dataset_command == "scan":
        print(json.dumps(scan_source_manifest(args.source), ensure_ascii=False, indent=2))
        return 0
    if args.dataset_command == "inspect":
        result = inspect_dataset(
            source=args.source,
            format_name=str(args.format),
            hash_workers=int(args.hash_workers),
            max_files=args.max_files,
            max_samples=args.max_samples,
        )
        _print_inspect_result(result, json_output=bool(args.json))
        return 0
    if args.dataset_command == "verify":
        result = verify_dataset(
            source=args.source,
            format_name=str(args.format),
            hash_workers=int(args.hash_workers),
            max_files=args.max_files,
            max_samples=args.max_samples,
        )
        _print_verify_result(result, json_output=bool(args.json))
        return 0 if str(result.get("status")) == "ok" else 1
    api = api_from_args(args)
    token = token_from_args(args, api=api)
    if args.dataset_command == "cleanup-plan":
        result = build_cleanup_plan(
            api=api,
            token=token,
            project_id=str(args.project),
            source=args.source,
            state_dir=args.state_dir,
            limit=int(args.limit),
        )
        _print_cleanup_plan(result, json_output=bool(args.json))
        return 0
    if args.dataset_command == "upload":
        result = upload_dataset(args, api=api, token=token)
        _print_upload_result(result, json_output=bool(args.json))
        return 0
    if args.dataset_command == "upload-archive":
        result = dataset_upload_archive(args, api=api, token=token)
        _print_archive_result(result, json_output=bool(args.json))
        return 0
    raise RuntimeError(f"unsupported dataset command: {args.dataset_command}")


def _print_upload_result(result: dict[str, object], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    job = result.get("job")
    complete = result.get("complete")
    status_source = job if isinstance(job, dict) else complete if isinstance(complete, dict) else {}
    status = str(status_source.get("status", "queued"))
    print(
        f"uploaded {result['file_count']} files to project {result['project_id']} "
        f"(import_id={result['import_id']}, status={status})"
    )


def _print_inspect_result(result: dict[str, object], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(
        f"{result['format']} dataset: {result['file_count']} files, "
        f"{result['image_count']} images, {result['label_count']} labels, "
        f"{result['total_bytes']} bytes"
    )


def _print_verify_result(result: dict[str, object], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(
        f"dataset verify {result['status']}: "
        f"{result['error_count']} errors, {result['warning_count']} warnings"
    )


def _print_cleanup_plan(result: dict[str, object], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    actions = list(result.get("actions") or [])
    server_imports = list(result.get("server_imports") or [])
    local_states = list(result.get("local_states") or [])
    print(
        f"cleanup plan: {len(actions)} actions, "
        f"{len(local_states)} local states, {len(server_imports)} server imports"
    )


def _print_archive_result(result: dict[str, object], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    archive = result.get("archive")
    archive_path = archive.get("archive_path") if isinstance(archive, dict) else ""
    job = result.get("job")
    complete = result.get("complete")
    status_source = job if isinstance(job, dict) else complete if isinstance(complete, dict) else {}
    status = str(status_source.get("status", "queued"))
    print(
        f"uploaded archive {archive_path} to project {result['project_id']} "
        f"(import_id={result['import_id']}, status={status})"
    )
