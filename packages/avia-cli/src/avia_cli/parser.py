from __future__ import annotations

import argparse
import os

_SUPPORTED_FORMATS = ("yolo", "coco", "imagenet", "anomalib")
_SUPPORTED_TASK_KEYS = ("detect", "classify", "segment", "pose", "obb", "ad")
_MAX_FOLDER_BATCH_SIZE = 1000
DEFAULT_STREAMING_BATCH_HTTP_TIMEOUT_SECONDS = 180.0
_DEFAULT_UPLOAD_READ_TIMEOUT = 45.0
_DEFAULT_UPLOAD_RETRY_BASE_DELAY = 0.25
_DEFAULT_API_BASE = "https://avia.eurekailab.com/api/v1"
_DEFAULT_HASH_WORKERS = min(8, os.cpu_count() or 1)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _integer_range(minimum: int, maximum: int):
    def parse_value(value: str) -> int:
        parsed = int(value)
        if parsed < minimum or parsed > maximum:
            raise argparse.ArgumentTypeError(f"must be between {minimum} and {maximum}")
        return parsed

    return parse_value


def _add_dataset_contract_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--format",
        required=True,
        choices=_SUPPORTED_FORMATS,
        help="Dataset serialization format; never inferred.",
    )
    parser.add_argument(
        "--task-key",
        required=True,
        choices=_SUPPORTED_TASK_KEYS,
        help=(
            "Exact matrix: yolo=detect,classify,segment,pose,obb; "
            "coco=detect,segment,pose; imagenet=classify; anomalib=ad."
        ),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="avia")
    sub = parser.add_subparsers(dest="command", required=True)

    auth_parser = sub.add_parser("auth")
    auth_sub = auth_parser.add_subparsers(dest="auth_command", required=True)

    login_parser = auth_sub.add_parser("login")
    login_parser.add_argument("--api", default=os.environ.get("AVIA_API_BASE", _DEFAULT_API_BASE))
    login_parser.add_argument("--token", default="")
    login_parser.add_argument("--token-stdin", action="store_true")
    login_parser.add_argument("--no-browser", action="store_true")
    login_parser.add_argument("--device-timeout", type=_positive_int, default=600)
    login_parser.add_argument("--poll-interval", type=_positive_float, default=None)

    status_parser = auth_sub.add_parser("status")
    status_parser.add_argument("--json", action="store_true")

    auth_sub.add_parser("logout")

    import_parser = sub.add_parser("import")
    import_sub = import_parser.add_subparsers(dest="import_command", required=True)

    scan_parser = import_sub.add_parser("scan")
    scan_parser.add_argument("--source", required=True)
    _add_dataset_contract_arguments(scan_parser)

    create_parser = import_sub.add_parser("create")
    create_parser.add_argument("--api", default=None)
    create_parser.add_argument("--token", default=None)
    create_parser.add_argument("--project", required=True)
    create_parser.add_argument("--source", required=True)
    create_parser.add_argument("--source-kind", default="object_prefix", choices=["object_prefix"])
    _add_dataset_contract_arguments(create_parser)
    create_parser.add_argument("--class", dest="class_name", action="append", default=[])
    create_parser.add_argument(
        "--no-auto-post-processing",
        dest="auto_post_processing",
        action="store_false",
        default=True,
    )

    dataset_parser = sub.add_parser("dataset")
    dataset_sub = dataset_parser.add_subparsers(dest="dataset_command", required=True)

    dataset_scan_parser = dataset_sub.add_parser("scan")
    dataset_scan_parser.add_argument("--source", required=True)
    _add_dataset_contract_arguments(dataset_scan_parser)

    inspect_parser = dataset_sub.add_parser("inspect")
    inspect_parser.add_argument("--source", required=True)
    _add_dataset_contract_arguments(inspect_parser)
    inspect_parser.add_argument(
        "--hash-workers",
        type=_positive_int,
        default=_DEFAULT_HASH_WORKERS,
    )
    inspect_parser.add_argument("--json", action="store_true")

    verify_parser = dataset_sub.add_parser("verify")
    verify_parser.add_argument("--source", required=True)
    _add_dataset_contract_arguments(verify_parser)
    verify_parser.add_argument(
        "--hash-workers",
        type=_positive_int,
        default=_DEFAULT_HASH_WORKERS,
    )
    verify_parser.add_argument("--json", action="store_true")

    cleanup_parser = dataset_sub.add_parser("cleanup-plan")
    cleanup_parser.add_argument("--api", default=None)
    cleanup_parser.add_argument("--token", default=None)
    cleanup_parser.add_argument("--project", required=True)
    cleanup_parser.add_argument("--source", default=None)
    cleanup_parser.add_argument("--state-dir", default=None)
    cleanup_parser.add_argument("--limit", type=_integer_range(1, 200), default=50)
    cleanup_parser.add_argument("--json", action="store_true")

    upload_parser = dataset_sub.add_parser("upload")
    upload_parser.add_argument("--api", default=None)
    upload_parser.add_argument("--token", default=None)
    upload_parser.add_argument("--project", required=True)
    upload_parser.add_argument("--source", required=True)
    _add_dataset_contract_arguments(upload_parser)
    upload_parser.add_argument("--class", dest="class_name", action="append", default=[])
    upload_parser.add_argument(
        "--concurrency",
        type=_positive_int,
        default=None,
        help="PUT upload concurrency (default: auto by hardware/network).",
    )
    upload_parser.add_argument(
        "--batch-size",
        type=_integer_range(1, _MAX_FOLDER_BATCH_SIZE),
        default=None,
        help="Files per upload batch (default: auto by hardware/network).",
    )
    upload_parser.add_argument("--batch-upload-url-timeout", type=_positive_float, default=60.0)
    upload_parser.add_argument("--batch-upload-url-retries", type=_positive_int, default=3)
    upload_parser.add_argument(
        "--batch-complete-timeout",
        type=_positive_float,
        default=DEFAULT_STREAMING_BATCH_HTTP_TIMEOUT_SECONDS,
    )
    upload_parser.add_argument("--batch-complete-retries", type=_positive_int, default=4)
    upload_parser.add_argument(
        "--batch-complete-concurrency",
        type=_positive_int,
        default=None,
        help="Batch-complete concurrency (default: auto by hardware/network).",
    )
    upload_parser.add_argument(
        "--stream-flush-size",
        type=_positive_int,
        default=None,
        help="Files buffered before a stream-complete batch (default: auto by hardware/network).",
    )
    upload_parser.add_argument(
        "--hash-workers",
        type=_positive_int,
        default=None,
        help="SHA-256 hashing workers (default: auto by hardware/network).",
    )
    upload_parser.add_argument("--state-flush-every", type=_positive_int, default=200)
    upload_parser.add_argument("--state-flush-interval", type=_positive_float, default=2.0)
    upload_parser.add_argument("--progress-interval", type=_positive_float, default=5.0)
    upload_parser.add_argument("--upload-retries", type=_positive_int, default=2)
    upload_parser.add_argument(
        "--upload-retry-base-delay",
        type=_positive_float,
        default=_DEFAULT_UPLOAD_RETRY_BASE_DELAY,
    )
    upload_parser.add_argument("--upload-connect-timeout", type=_positive_float, default=15.0)
    upload_parser.add_argument(
        "--upload-read-timeout",
        type=_positive_float,
        default=_DEFAULT_UPLOAD_READ_TIMEOUT,
    )
    upload_parser.add_argument("--resume", action="store_true")
    upload_parser.add_argument("--state-dir", default=None)
    upload_parser.add_argument("--wait", action="store_true")
    upload_parser.add_argument("--wait-timeout", type=_positive_int, default=600)
    upload_parser.add_argument("--poll-interval", type=_positive_float, default=2.0)
    upload_parser.add_argument("--json", action="store_true")

    return parser
