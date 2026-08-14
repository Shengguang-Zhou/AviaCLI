from __future__ import annotations

import json

from avia_cli.context import api_from_args, token_from_args
from avia_cli.core.uploads.contracts import require_format_task
from avia_cli.core.uploads.dataset import create_source_import
from avia_cli.core.uploads.state import _source_import_payload


def handle_import_command(args) -> int:
    require_format_task(format_name=args.format, task_key=args.task_key)
    if args.import_command == "create":
        payload = _source_import_payload(args)
        api = api_from_args(args)
        token = token_from_args(args, api=api)
        result = create_source_import(
            api=api,
            token=token,
            project_id=str(args.project),
            payload=payload,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    raise RuntimeError(f"unsupported import command: {args.import_command}")
