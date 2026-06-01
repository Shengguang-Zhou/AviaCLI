from __future__ import annotations

import sys

from avia_cli.commands.auth import handle_auth_command
from avia_cli.commands.dataset import handle_dataset_command
from avia_cli.commands.imports import handle_import_command
from avia_cli.parser import _build_parser
from avia_cli.core import errors


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args(sys.argv[1:])
    try:
        if args.command == "auth":
            return handle_auth_command(args)
        if args.command == "import":
            return handle_import_command(args)
        if args.command == "dataset":
            return handle_dataset_command(args)
    except (errors._AviaHTTPError, RuntimeError) as exc:
        message = (
            errors.format_avia_http_error(exc)
            if isinstance(exc, errors._AviaHTTPError)
            else f"Error: {exc}"
        )
        print(message, file=sys.stderr)
        return 1
    parser.error("unsupported command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
