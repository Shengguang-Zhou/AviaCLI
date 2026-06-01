from __future__ import annotations

import json

from avia_cli.context import api_from_args, token_from_args
from avia_sdk.uploads.archive import dataset_upload_archive
from avia_sdk.uploads.dataset import upload_dataset
from avia_sdk.uploads.manifest import scan_source_manifest


def handle_dataset_command(args) -> int:
    if args.dataset_command == "scan":
        print(json.dumps(scan_source_manifest(args.source), ensure_ascii=False, indent=2))
        return 0
    api = api_from_args(args)
    token = token_from_args(args, api=api)
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
