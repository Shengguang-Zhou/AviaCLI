from __future__ import annotations

import json

from avia_cli.context import api_from_args, token_from_args
from avia_cli.core.uploads.contracts import require_format_task
from avia_cli.core.uploads.dataset import prepare_dataset_upload, upload_prepared_dataset
from avia_cli.core.uploads.inspect import (
    build_cleanup_plan,
    inspect_dataset,
    verify_dataset,
)
from avia_cli.core.uploads.manifest import scan_source_manifest


def handle_dataset_command(args) -> int:
    if args.dataset_command in {"scan", "inspect", "verify", "upload"}:
        require_format_task(format_name=str(args.format), task_key=str(args.task_key))
    if args.dataset_command == "scan":
        result = scan_source_manifest(args.source, format_name=str(args.format))
        result["format"] = str(args.format)
        result["task_key"] = str(args.task_key)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.dataset_command == "inspect":
        result = inspect_dataset(
            source=args.source,
            format_name=str(args.format),
            task_key=str(args.task_key),
            hash_workers=int(args.hash_workers),
        )
        _print_inspect_result(result, json_output=bool(args.json))
        return 0
    if args.dataset_command == "verify":
        result = verify_dataset(
            source=args.source,
            format_name=str(args.format),
            task_key=str(args.task_key),
            hash_workers=int(args.hash_workers),
        )
        _print_verify_result(result, json_output=bool(args.json))
        return 0 if str(result.get("status")) == "ok" else 1
    prepared_upload = prepare_dataset_upload(args) if args.dataset_command == "upload" else None
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
        if prepared_upload is None:
            raise RuntimeError("dataset upload preparation is missing")
        result = upload_prepared_dataset(
            args,
            api=api,
            token=token,
            prepared=prepared_upload,
        )
        _print_upload_result(result, json_output=bool(args.json))
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
        f"{result['format']}/{result['task_key']} dataset: {result['file_count']} files, "
        f"{result['image_count']} images, {result['label_count']} labels, "
        f"{result['total_bytes']} bytes"
    )


def _print_verify_result(result: dict[str, object], *, json_output: bool) -> None:
    if json_output:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(
        f"dataset verify {result['format']}/{result['task_key']} {result['status']}: "
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
