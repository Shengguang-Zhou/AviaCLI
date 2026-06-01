from __future__ import annotations

import argparse
import os

_SUPPORTED_FORMATS = ("yolo", "coco", "imagenet")
_MAX_FOLDER_BATCH_SIZE = 1000
DEFAULT_STREAMING_BATCH_HTTP_TIMEOUT_SECONDS = 180.0
_DEFAULT_UPLOAD_READ_TIMEOUT = 45.0
_DEFAULT_UPLOAD_RETRY_BASE_DELAY = 0.25
_DEFAULT_API_BASE = "https://avia.eurekailab.com/api/v1"


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
    login_parser.add_argument("--device-timeout", type=int, default=600)
    login_parser.add_argument("--poll-interval", type=float, default=None)

    status_parser = auth_sub.add_parser("status")
    status_parser.add_argument("--json", action="store_true")

    auth_sub.add_parser("logout")

    import_parser = sub.add_parser("import")
    import_sub = import_parser.add_subparsers(dest="import_command", required=True)

    scan_parser = import_sub.add_parser("scan")
    scan_parser.add_argument("--source", required=True)

    create_parser = import_sub.add_parser("create")
    create_parser.add_argument("--api", default=None)
    create_parser.add_argument("--token", default=None)
    create_parser.add_argument("--project", required=True)
    create_parser.add_argument("--source", required=True)
    create_parser.add_argument(
        "--source-kind", default="object_prefix", choices=["object_prefix"]
    )
    create_parser.add_argument("--format", default="yolo", choices=_SUPPORTED_FORMATS)
    create_parser.add_argument("--task-key", default="detect")
    create_parser.add_argument(
        "--class", dest="class_name", action="append", default=[]
    )
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

    upload_parser = dataset_sub.add_parser("upload")
    upload_parser.add_argument("--api", default=None)
    upload_parser.add_argument("--token", default=None)
    upload_parser.add_argument("--project", required=True)
    upload_parser.add_argument("--source", required=True)
    upload_parser.add_argument("--format", default="yolo", choices=_SUPPORTED_FORMATS)
    upload_parser.add_argument("--task-key", default="detect")
    upload_parser.add_argument(
        "--class", dest="class_name", action="append", default=[]
    )
    upload_parser.add_argument("--concurrency", type=int, default=32)
    upload_parser.add_argument("--batch-size", type=int, default=1000)
    upload_parser.add_argument("--max-files", type=int, default=None)
    upload_parser.add_argument("--max-samples", type=int, default=None)
    upload_parser.add_argument("--batch-upload-url-timeout", type=float, default=60.0)
    upload_parser.add_argument("--batch-upload-url-retries", type=int, default=3)
    upload_parser.add_argument(
        "--batch-complete-timeout",
        type=float,
        default=DEFAULT_STREAMING_BATCH_HTTP_TIMEOUT_SECONDS,
    )
    upload_parser.add_argument("--batch-complete-retries", type=int, default=4)
    upload_parser.add_argument("--batch-complete-concurrency", type=int, default=1)
    upload_parser.add_argument("--stream-flush-size", type=int, default=512)
    upload_parser.add_argument(
        "--auto-crop-embedding",
        dest="auto_crop_embedding",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    upload_parser.add_argument(
        "--hash-workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
    )
    upload_parser.add_argument("--state-flush-every", type=int, default=200)
    upload_parser.add_argument("--state-flush-interval", type=float, default=2.0)
    upload_parser.add_argument("--progress-interval", type=float, default=5.0)
    upload_parser.add_argument("--upload-retries", type=int, default=2)
    upload_parser.add_argument(
        "--upload-retry-base-delay",
        type=float,
        default=_DEFAULT_UPLOAD_RETRY_BASE_DELAY,
    )
    upload_parser.add_argument("--upload-connect-timeout", type=float, default=15.0)
    upload_parser.add_argument(
        "--upload-read-timeout",
        type=float,
        default=_DEFAULT_UPLOAD_READ_TIMEOUT,
    )
    upload_parser.add_argument(
        "--upload-url-origin-override",
        default="",
        help=(
            "Connect to this http(s) origin for direct object-storage presigned URLs while "
            "preserving the signed Host header. Intended for server-local benchmarks."
        ),
    )
    upload_parser.add_argument("--resume", action="store_true")
    upload_parser.add_argument("--state-dir", default=None)
    upload_parser.add_argument("--wait", action="store_true")
    upload_parser.add_argument("--wait-timeout", type=int, default=600)
    upload_parser.add_argument("--poll-interval", type=float, default=2.0)
    upload_parser.add_argument("--json", action="store_true")

    archive_parser = dataset_sub.add_parser("upload-archive")
    archive_parser.add_argument("--api", default=None)
    archive_parser.add_argument("--token", default=None)
    archive_parser.add_argument("--project", required=True)
    archive_parser.add_argument("--source", required=True)
    archive_parser.add_argument("--format", default="yolo", choices=_SUPPORTED_FORMATS)
    archive_parser.add_argument(
        "--transport",
        default="object-storage",
        choices=["object-storage"],
    )
    archive_parser.add_argument("--archive-path", default=None)
    archive_parser.add_argument("--force-archive", action="store_true")
    archive_parser.add_argument("--expires-in", type=int, default=3600)
    archive_parser.add_argument(
        "--multipart-part-size-mb",
        type=int,
        default=0,
        help="Multipart part size in MiB. Use 0 for auto sizing.",
    )
    archive_parser.add_argument("--multipart-concurrency", type=int, default=32)
    archive_parser.add_argument("--multipart-sign-batch-size", type=int, default=256)
    archive_parser.add_argument("--progress-interval", type=float, default=5.0)
    archive_parser.add_argument("--upload-retries", type=int, default=5)
    archive_parser.add_argument("--upload-retry-base-delay", type=float, default=1.0)
    archive_parser.add_argument("--wait", action="store_true")
    archive_parser.add_argument("--wait-timeout", type=int, default=7200)
    archive_parser.add_argument("--poll-interval", type=float, default=5.0)
    archive_parser.add_argument("--json", action="store_true")
    return parser
