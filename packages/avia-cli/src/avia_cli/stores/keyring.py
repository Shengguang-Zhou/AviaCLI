from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from platformdirs import user_config_path
from filelock import FileLock

from avia_cli.core.api_base import canonical_api_base
from avia_cli.core.atomic_file import durable_atomic_write, read_regular_file

_SERVICE_NAME = "avia-cli"
_DEFAULT_PROFILE = "default"
_CONFIG_FIELDS = {"active_profile", "profiles"}
_PROFILE_FIELDS = {
    "api",
    "key_prefix",
    "keyring_username",
    "role",
    "saved_at",
    "user_id",
    "workspace_id",
}


@dataclass(frozen=True)
class CliAuthProfile:
    profile: str
    api: str
    token: str
    workspace_id: str
    user_id: str
    role: str
    key_prefix: str


def _config_dir() -> Path:
    configured = str(os.environ.get("AVIA_CLI_CONFIG_DIR") or "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path(user_config_path("avia", "Eurekai"))


def _config_path() -> Path:
    return _config_dir() / "config.json"


def _config_lock() -> FileLock:
    _config_dir().mkdir(parents=True, exist_ok=True)
    os.chmod(_config_dir(), 0o700)
    return FileLock(_config_dir() / "config.lock", timeout=10)


def _keyring_username(*, profile: str, api: str) -> str:
    return f"{profile}:{api.rstrip('/')}"


def _load_keyring():
    try:
        return importlib.import_module("keyring")
    except Exception as exc:  # pragma: no cover - import failure shape depends on host
        raise RuntimeError(
            "System keyring is required for `avia auth login`. "
            "Install keyring support or use AVIA_TOKEN for server/CI uploads."
        ) from exc


def _read_config() -> dict[str, Any]:
    path = _config_path()
    if not path.exists():
        return {}
    try:
        config = json.loads(read_regular_file(path).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid Avia CLI config: {path}") from exc
    if not isinstance(config, dict) or set(config) != _CONFIG_FIELDS:
        raise RuntimeError(f"Invalid Avia CLI config fields: {path}")
    active = config.get("active_profile")
    profiles = config.get("profiles")
    if (
        not isinstance(active, str)
        or not active
        or active != active.strip()
        or not isinstance(profiles, dict)
        or not profiles
    ):
        raise RuntimeError(f"Invalid Avia CLI config profile index: {path}")
    if active not in profiles:
        raise RuntimeError(f"Active Avia CLI profile is missing: {path}")
    for name, metadata in profiles.items():
        if (
            not isinstance(name, str)
            or not name
            or name != name.strip()
            or not isinstance(metadata, dict)
        ):
            raise RuntimeError(f"Invalid Avia CLI profile entry: {path}")
        if set(metadata) != _PROFILE_FIELDS:
            raise RuntimeError(f"Invalid Avia CLI profile fields for {name}: {path}")
        try:
            api = canonical_api_base(metadata.get("api"))
        except ValueError as exc:
            raise RuntimeError(f"Invalid Avia CLI profile API for {name}: {path}") from exc
        for field in _PROFILE_FIELDS - {"api", "saved_at"}:
            value = metadata.get(field)
            if not isinstance(value, str) or not value or value != value.strip():
                raise RuntimeError(f"Invalid Avia CLI profile {field} for {name}: {path}")
        if metadata["keyring_username"] != _keyring_username(profile=name, api=api):
            raise RuntimeError(f"Invalid Avia CLI profile keyring identity for {name}: {path}")
        saved_at = metadata.get("saved_at")
        if not isinstance(saved_at, str):
            raise RuntimeError(f"Invalid Avia CLI profile saved_at for {name}: {path}")
        try:
            parsed_saved_at = datetime.fromisoformat(saved_at)
        except ValueError as exc:
            raise RuntimeError(f"Invalid Avia CLI profile saved_at for {name}: {path}") from exc
        if parsed_saved_at.utcoffset() is None:
            raise RuntimeError(f"Invalid Avia CLI profile saved_at timezone for {name}: {path}")
    return config


def _write_config(config: dict[str, Any]) -> None:
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    durable_atomic_write(
        path,
        (json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        mode=0o600,
    )


def save_cli_auth_profile(
    *,
    api: str,
    token: str,
    workspace_id: str,
    user_id: str,
    role: str,
    profile: str = _DEFAULT_PROFILE,
) -> CliAuthProfile:
    try:
        clean_api = canonical_api_base(api)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    raw_token = str(token or "")
    if not raw_token or raw_token != raw_token.strip():
        raise RuntimeError("API token is required for CLI auth login.")
    clean_token = raw_token
    identity = {
        "profile": profile,
        "workspace_id": workspace_id,
        "user_id": user_id,
        "role": role,
    }
    for field, raw_value in identity.items():
        value = str(raw_value or "")
        if not value or value != value.strip():
            raise RuntimeError(f"{field} must be a canonical non-empty string.")

    username = _keyring_username(profile=profile, api=clean_api)
    with _config_lock():
        keyring = _load_keyring()
        previous = keyring.get_password(_SERVICE_NAME, username)
        keyring.set_password(_SERVICE_NAME, username, clean_token)
        try:
            config = _read_config()
            profiles = dict(config.get("profiles") or {})
            profiles[profile] = {
                "api": clean_api,
                "workspace_id": str(workspace_id),
                "user_id": str(user_id),
                "role": str(role),
                "key_prefix": clean_token[:12],
                "keyring_username": username,
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            config = {"active_profile": profile, "profiles": profiles}
            _write_config(config)
        except BaseException:
            if previous is None:
                keyring.delete_password(_SERVICE_NAME, username)
            else:
                keyring.set_password(_SERVICE_NAME, username, previous)
            raise
    return CliAuthProfile(
        profile=profile,
        api=clean_api,
        token=clean_token,
        workspace_id=str(workspace_id),
        user_id=str(user_id),
        role=str(role),
        key_prefix=clean_token[:12],
    )


def load_cli_auth_profile(profile: str | None = None) -> CliAuthProfile | None:
    config = _read_config()
    profiles = dict(config.get("profiles") or {})
    selected = str(profile or config.get("active_profile") or _DEFAULT_PROFILE)
    meta = profiles.get(selected)
    if not isinstance(meta, dict):
        return None
    api = canonical_api_base(meta.get("api"))
    username = str(meta["keyring_username"])
    token = str(_load_keyring().get_password(_SERVICE_NAME, username) or "").strip()
    if not token:
        raise RuntimeError(
            f"Avia CLI auth profile `{selected}` is missing its keyring token. "
            "Run `avia auth login --api ...` again or set AVIA_TOKEN."
        )
    return CliAuthProfile(
        profile=selected,
        api=api,
        token=token,
        workspace_id=str(meta["workspace_id"]),
        user_id=str(meta["user_id"]),
        role=str(meta["role"]),
        key_prefix=str(meta["key_prefix"]),
    )


def clear_cli_auth_profile(profile: str | None = None) -> bool:
    with _config_lock():
        config = _read_config()
        profiles = dict(config.get("profiles") or {})
        selected = str(profile or config.get("active_profile") or _DEFAULT_PROFILE)
        meta = profiles.get(selected)
        if not isinstance(meta, dict):
            return False
        username = str(meta["keyring_username"])
        keyring = _load_keyring()
        token = keyring.get_password(_SERVICE_NAME, username)
        profiles.pop(selected)
        path = _config_path()
        if profiles:
            next_config = {"profiles": profiles, "active_profile": next(iter(profiles))}
            _write_config(next_config)
        elif path.exists():
            path.unlink()
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_CLOEXEC)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        try:
            keyring.delete_password(_SERVICE_NAME, username)
        except BaseException:
            if token is not None:
                keyring.set_password(_SERVICE_NAME, username, token)
            _write_config(config)
            raise
        return True
