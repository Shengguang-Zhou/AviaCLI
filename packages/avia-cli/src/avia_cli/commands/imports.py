from __future__ import annotations

import json

from avia_cli.context import api_from_args, token_from_args
from avia_cli.core.uploads.dataset import create_source_import
from avia_cli.core.uploads.manifest import scan_source_manifest
from avia_cli.core.uploads.state import _source_import_payload


def handle_import_command(args) -> int:
    if args.import_command == "scan":
        print(json.dumps(scan_source_manifest(args.source), ensure_ascii=False, indent=2))
        return 0
    if args.import_command == "create":
        api = api_from_args(args)
        token = token_from_args(args, api=api)
        result = create_source_import(
            api=api,
            token=token,
            project_id=str(args.project),
            payload=_source_import_payload(args),
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    raise RuntimeError(f"unsupported import command: {args.import_command}")
