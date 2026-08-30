from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import uuid
from contextlib import contextmanager
from collections.abc import Callable
from pathlib import Path
from typing import Any, Iterable, Iterator

from filelock import FileLock, Timeout

from avia_cli.core.api_base import canonical_api_base
from avia_cli.core.uploads.contracts import ANOMALIB_CLASSES, require_object_prefix_uri
from avia_cli.core.uploads.inventory import is_dataset_image_path
from avia_cli.core.uploads.media_types import require_canonical_media_type
from avia_cli.core.uploads.response_contracts import validate_source_import_request
from avia_cli.core.atomic_file import durable_atomic_write, read_regular_file
from avia_cli.core.uploads.response_contracts import decode_complete_import_response
from avia_cli.core.uploads.source_file import SourceIdentity, open_verified_source

_SESSION_PAYLOAD_FIELDS = {
    "idempotency_key",
    "format",
    "root_name",
    "task_keys",
    "classes",
    "file_count",
    "total_bytes",
}
_STATE_FIELDS = {
    "api",
    "complete_response",
    "files",
    "format",
    "idempotency_key",
    "import_id",
    "phase",
    "project_id",
    "schema_version",
    "session_payload",
    "source",
    "task_key",
}
_STATE_FILE_FIELDS = {
    "content_type",
    "height",
    "object_key",
    "sha256",
    "size_bytes",
    "source_identity",
    "streamed",
    "uploaded",
    "version_id",
    "width",
}


def _source_import_payload(args: argparse.Namespace) -> dict[str, object]:
    format_name = str(args.format)
    requested_classes = list(args.class_name or [])
    if requested_classes and format_name != "yolo":
        raise SystemExit("--class is only valid with --format yolo")
    classes = list(ANOMALIB_CLASSES) if format_name == "anomalib" else requested_classes
    payload: dict[str, object] = {
        "source_kind": str(args.source_kind),
        "uri": require_object_prefix_uri(args.source),
        "format": format_name,
        "task_key": str(args.task_key),
        "classes": classes,
        "auto_post_processing": bool(args.auto_post_processing),
    }
    validate_source_import_request(payload)
    return payload


def _chunked(items: list[dict[str, object]], size: int) -> Iterable[list[dict[str, object]]]:
    step = int(size)
    if step <= 0:
        raise ValueError("chunk size must be greater than zero")
    for idx in range(0, len(items), step):
        yield items[idx : idx + step]


def _item_with_upload_metadata(
    source_root: Path,
    item: dict[str, object],
    *,
    source_identity: SourceIdentity,
    is_image_path: Callable[[str], bool] = is_dataset_image_path,
) -> dict[str, object]:
    enriched = dict(item)
    relative_path = str(item["relative_path"])
    path = source_root / relative_path
    needs_sha256 = not str(enriched.get("sha256") or "").strip()
    needs_dimensions = is_image_path(relative_path) and (
        not int(enriched.get("width") or 0) or not int(enriched.get("height") or 0)
    )
    with open_verified_source(path, source_identity) as source:
        if needs_sha256:
            digest = hashlib.sha256()
            for chunk in iter(lambda: source.handle.read(1024 * 1024), b""):
                digest.update(chunk)
            enriched["sha256"] = digest.hexdigest()
        if needs_dimensions:
            from PIL import Image

            source.handle.seek(0)
            with Image.open(source.handle) as image:
                image.verify()
            source.handle.seek(0)
            with Image.open(source.handle) as image:
                image.load()
                width, height = int(image.width), int(image.height)
            if width <= 0 or height <= 0:
                raise ValueError(f"image dimensions must be positive: {path}")
            enriched["width"] = width
            enriched["height"] = height
        source.assert_unchanged(context="source file changed while computing upload metadata")
    return enriched


def _ensure_sha256_batch(
    *,
    source_root: Path,
    files: list[dict[str, object]],
    hash_workers: int,
    source_identities: dict[str, SourceIdentity],
    is_image_path: Callable[[str], bool] = is_dataset_image_path,
) -> list[dict[str, object]]:
    missing = [
        item
        for item in files
        if not str(item.get("sha256") or "").strip()
        or (
            is_image_path(str(item["relative_path"]))
            and (not int(item.get("width") or 0) or not int(item.get("height") or 0))
        )
    ]
    if not missing:
        return files

    workers = int(hash_workers)
    if workers <= 0:
        raise ValueError("hash_workers must be greater than zero")
    hashed_by_relative: dict[str, dict[str, object]] = {}
    if workers == 1 or len(missing) == 1:
        for item in missing:
            hashed = _item_with_upload_metadata(
                source_root,
                item,
                source_identity=source_identities[str(item["relative_path"])],
                is_image_path=is_image_path,
            )
            hashed_by_relative[str(hashed["relative_path"])] = hashed
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            for hashed in executor.map(
                lambda item: _item_with_upload_metadata(
                    source_root,
                    item,
                    source_identity=source_identities[str(item["relative_path"])],
                    is_image_path=is_image_path,
                ),
                missing,
            ):
                hashed_by_relative[str(hashed["relative_path"])] = hashed

    return [hashed_by_relative.get(str(item["relative_path"]), item) for item in files]


def _safe_state_segment(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)
    return cleaned.strip("._") or "default"


def _state_dir(args: argparse.Namespace) -> Path:
    explicit = str(getattr(args, "state_dir", "") or os.environ.get("AVIA_STATE_DIR", "")).strip()
    if explicit:
        return Path(explicit).expanduser().resolve()
    return Path(str(args.source)).expanduser().resolve() / ".avia" / "imports"


def _state_path(state_dir: Path, project_id: str, idempotency_key: str) -> Path:
    return state_dir / _safe_state_segment(project_id) / f"{idempotency_key}.json"


@contextmanager
def _exclusive_upload_state_lock(
    *,
    state_dir: Path,
    project_id: str,
    source: str,
    import_format: str,
    task_key: str,
) -> Iterator[Path]:
    identity = {
        "format": str(import_format),
        "project_id": str(project_id),
        "source": str(source),
        "task_key": str(task_key),
    }
    digest = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
    ).hexdigest()
    project_dir = state_dir / _safe_state_segment(project_id)
    project_dir.mkdir(parents=True, exist_ok=True)
    guard_path = project_dir / f"{digest}.guard"
    owner_path = project_dir / f"{digest}.owner"
    lock = FileLock(guard_path, timeout=0)
    try:
        lock.acquire()
    except Timeout as exc:
        try:
            owner = owner_path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            owner = "<owner metadata not yet published>"
        except OSError as owner_exc:
            owner = f"<owner metadata unreadable: {owner_exc}>"
        raise SystemExit(
            f"upload already active for this dataset; owner={owner} lock={guard_path}"
        ) from exc

    record = {**identity, "pid": os.getpid()}
    try:
        durable_atomic_write(
            owner_path,
            json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8"),
            mode=0o600,
        )
        yield owner_path
    finally:
        owner_path.unlink(missing_ok=True)
        lock.release()


def _save_state(state_dir: Path, state: dict[str, Any]) -> None:
    path = _state_path(
        state_dir,
        str(state.get("project_id") or "project"),
        _require_idempotency_key(state.get("idempotency_key")),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    _validate_state(state, path=path)
    durable_atomic_write(
        path,
        json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8"),
        mode=0o600,
    )


def _load_resume_state(
    *,
    state_dir: Path,
    project_id: str,
    api: str,
    source: str,
    import_format: str,
    task_key: str,
) -> dict[str, Any] | None:
    project_dir = state_dir / _safe_state_segment(project_id)
    if not project_dir.exists():
        return None
    candidates = sorted(project_dir.glob("*.json"))
    validated: list[tuple[Path, dict[str, Any]]] = []
    for path in candidates:
        try:
            state = json.loads(read_regular_file(path).decode("utf-8"))
        except (OSError, RuntimeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SystemExit(f"invalid resume state {path}: {exc}") from exc
        if not isinstance(state, dict):
            raise SystemExit(f"invalid resume state {path}: expected JSON object")
        try:
            _validate_state(state, path=path)
        except ValueError as exc:
            raise SystemExit(f"invalid resume state {path}: {exc}") from exc
        validated.append((path, state))

    near_matches = [
        (path, state)
        for path, state in validated
        if (
            str(state.get("source") or "") == str(source)
            and str(state.get("format") or "") == str(import_format)
            and str(state.get("project_id") or "") == str(project_id)
        )
    ]
    exact_matches = [
        (path, state)
        for path, state in near_matches
        if state["api"] == api and state["task_key"] == task_key
    ]
    active_exact_matches = [
        (path, state) for path, state in exact_matches if state["phase"] != "completed"
    ]
    if active_exact_matches:
        exact_matches = active_exact_matches
    if len(exact_matches) == 1:
        return exact_matches[0][1]
    if len(exact_matches) > 1:
        identities = [
            {"import_id": state["import_id"], "path": str(path)} for path, state in exact_matches
        ]
        raise SystemExit(
            "ambiguous resume state: multiple exact states match "
            f"project={project_id!r} source={source!r} format={import_format!r} "
            f"api={api!r} task_key={task_key!r}: {identities!r}"
        )
    if not near_matches:
        return None

    api_matches = [(path, state) for path, state in near_matches if state["api"] == api]
    task_matches = [(path, state) for path, state in near_matches if state["task_key"] == task_key]
    paths = [str(path) for path, _state in near_matches]
    if not api_matches:
        state_apis = sorted({str(state["api"]) for _path, state in near_matches})
        raise SystemExit(
            f"resume state API mismatch: state={state_apis!r}, requested={api!r}, paths={paths!r}"
        )
    if not task_matches:
        state_tasks = sorted({str(state["task_key"]) for _path, state in near_matches})
        raise SystemExit(
            "resume state task mismatch: "
            f"state={state_tasks!r}, requested={task_key!r}, paths={paths!r}"
        )
    identities = [
        {"api": state["api"], "task_key": state["task_key"], "path": str(path)}
        for path, state in near_matches
    ]
    raise SystemExit(
        "resume state identity mismatch: API and task exist only on different states; "
        f"requested_api={api!r} requested_task={task_key!r} candidates={identities!r}"
    )


def _validate_state(state: dict[str, Any], *, path: Path) -> None:
    if set(state) != _STATE_FIELDS:
        raise ValueError("state fields must be exact")
    if state.get("schema_version") != 4:
        raise ValueError("state schema_version must be 4")
    phase = state.get("phase")
    if phase not in {"session_pending", "uploading", "completed"}:
        raise ValueError("state phase is invalid")
    import_id = state.get("import_id")
    if phase == "session_pending" and import_id is not None:
        raise ValueError("session_pending state must not have import_id")
    if phase != "session_pending" and (not isinstance(import_id, str) or not import_id):
        raise ValueError("uploading/completed state requires import_id")
    complete_response = state.get("complete_response")
    if phase == "completed":
        if not isinstance(complete_response, dict) or not complete_response:
            raise ValueError("completed state requires complete_response")
    elif complete_response is not None:
        raise ValueError("non-completed state must not have complete_response")
    for key in ("api", "project_id", "source", "format", "task_key"):
        value = state.get(key)
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"state {key} must be a canonical non-empty string")
    try:
        canonical_api_base(state["api"])
    except ValueError as exc:
        raise ValueError("state api is not canonical") from exc
    idempotency_key = _require_idempotency_key(state.get("idempotency_key"))
    if path.name != f"{idempotency_key}.json":
        raise ValueError("state filename must match idempotency_key")
    session_payload = state.get("session_payload")
    if not isinstance(session_payload, dict) or set(session_payload) != _SESSION_PAYLOAD_FIELDS:
        raise ValueError("session_payload fields must be exact")
    if session_payload.get("idempotency_key") != state.get("idempotency_key"):
        raise ValueError("session_payload idempotency_key mismatch")
    if phase == "completed":
        assert isinstance(complete_response, dict)
        try:
            decode_complete_import_response(
                complete_response,
                project_id=str(state["project_id"]),
                import_id=str(state["import_id"]),
            )
        except RuntimeError as exc:
            raise ValueError(f"completed state response is invalid: {exc}") from exc
    for key in ("format", "root_name"):
        value = session_payload.get(key)
        if not isinstance(value, str) or not value or value != value.strip():
            raise ValueError(f"session_payload {key} must be canonical")
    if session_payload.get("task_keys") != [state["task_key"]]:
        raise ValueError("session_payload task_keys must match the selected task")
    classes = session_payload.get("classes")
    if (
        not isinstance(classes, list)
        or any(not isinstance(item, str) or not item or item != item.strip() for item in classes)
        or len(set(classes)) != len(classes)
    ):
        raise ValueError("session_payload classes must be unique canonical strings")
    for key in ("file_count", "total_bytes"):
        value = session_payload.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"session_payload {key} must be a positive integer")
    files = state.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("state files must be a non-empty object")
    for relative_path, raw in files.items():
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError("state file paths must be non-empty strings")
        if not isinstance(raw, dict) or set(raw) != _STATE_FILE_FIELDS:
            raise ValueError(f"state file fields must be exact: {relative_path}")
        if not isinstance(raw.get("uploaded"), bool) or not isinstance(raw.get("streamed"), bool):
            raise ValueError(f"state file flags must be booleans: {relative_path}")
        if raw.get("streamed") and not raw.get("uploaded"):
            raise ValueError(f"streamed state file must also be uploaded: {relative_path}")
        for key in ("size_bytes", "width", "height"):
            value = raw.get(key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"state file {key} is invalid: {relative_path}")
        if bool(raw.get("width")) != bool(raw.get("height")):
            raise ValueError(f"state file dimensions must be paired: {relative_path}")
        sha256 = raw.get("sha256")
        if not isinstance(sha256, str) or (
            sha256 and (len(sha256) != 64 or any(item not in "0123456789abcdef" for item in sha256))
        ):
            raise ValueError(f"state file sha256 is invalid: {relative_path}")
        if raw.get("object_key") is not None and (
            not isinstance(raw.get("object_key"), str) or not raw.get("object_key")
        ):
            raise ValueError(f"state file object_key is invalid: {relative_path}")
        version_id = raw.get("version_id")
        if version_id is not None and (
            not isinstance(version_id, str)
            or not version_id
            or version_id != version_id.strip()
            or version_id == "null"
        ):
            raise ValueError(f"state file version_id is invalid: {relative_path}")
        content_type = raw.get("content_type")
        if content_type is not None:
            try:
                require_canonical_media_type(
                    content_type,
                    label=f"state file content_type for {relative_path}",
                )
            except RuntimeError as exc:
                raise ValueError(str(exc)) from exc
        identity = raw.get("source_identity")
        if not isinstance(identity, dict) or set(identity) != {
            "device",
            "inode",
            "size_bytes",
            "mtime_ns",
            "ctime_ns",
        }:
            raise ValueError(f"state file source_identity is invalid: {relative_path}")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in identity.values()
        ):
            raise ValueError(f"state file source_identity values are invalid: {relative_path}")
        if identity["size_bytes"] != raw["size_bytes"]:
            raise ValueError(f"state file source identity size mismatch: {relative_path}")
        if raw["uploaded"] and (
            not sha256
            or raw.get("object_key") is None
            or content_type is None
            or version_id is None
        ):
            raise ValueError(f"uploaded state file lacks remote identity: {relative_path}")
        if not raw["uploaded"] and (raw.get("object_key") is not None or content_type is not None):
            raise ValueError(f"non-uploaded state file has remote identity: {relative_path}")


def _require_idempotency_key(value: object) -> str:
    key = str(value or "")
    try:
        parsed = uuid.UUID(key)
    except (ValueError, AttributeError) as exc:
        raise ValueError("idempotency_key must be a canonical lowercase UUIDv4") from exc
    if parsed.version != 4 or str(parsed) != key:
        raise ValueError("idempotency_key must be a canonical lowercase UUIDv4")
    return key
