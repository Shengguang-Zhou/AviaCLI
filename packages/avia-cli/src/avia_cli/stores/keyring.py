from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from platformdirs import user_config_path

_SERVICE_NAME = "avia-cli"
_DEFAULT_PROFILE = "default"


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
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid Avia CLI config: {path}") from exc


def _write_config(config: dict[str, Any]) -> None:
    path = _config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    os.chmod(tmp, 0o600)
    tmp.replace(path)


def save_cli_auth_profile(
    *,
    api: str,
    token: str,
    workspace_id: str,
    user_id: str,
    role: str,
    profile: str = _DEFAULT_PROFILE,
) -> CliAuthProfile:
    clean_api = str(api or "").strip().rstrip("/")
    clean_token = str(token or "").strip()
    if not clean_api:
        raise RuntimeError("API base is required for CLI auth login.")
    if not clean_token:
        raise RuntimeError("API token is required for CLI auth login.")

    username = _keyring_username(profile=profile, api=clean_api)
    _load_keyring().set_password(_SERVICE_NAME, username, clean_token)
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
    config["active_profile"] = profile
    config["profiles"] = profiles
    _write_config(config)
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
    api = str(meta.get("api") or "").strip().rstrip("/")
    if not api:
        return None
    username = str(meta.get("keyring_username") or "").strip() or _keyring_username(
        profile=selected,
        api=api,
    )
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
        workspace_id=str(meta.get("workspace_id") or ""),
        user_id=str(meta.get("user_id") or ""),
        role=str(meta.get("role") or ""),
        key_prefix=str(meta.get("key_prefix") or token[:12]),
    )


def clear_cli_auth_profile(profile: str | None = None) -> bool:
    config = _read_config()
    profiles = dict(config.get("profiles") or {})
    selected = str(profile or config.get("active_profile") or _DEFAULT_PROFILE)
    meta = profiles.get(selected)
    if not isinstance(meta, dict):
        return False
    api = str(meta.get("api") or "").strip().rstrip("/")
    username = str(meta.get("keyring_username") or "").strip() or _keyring_username(
        profile=selected,
        api=api,
    )
    try:
        _load_keyring().delete_password(_SERVICE_NAME, username)
    except Exception:
        pass
    profiles.pop(selected, None)
    if profiles:
        config["profiles"] = profiles
        config["active_profile"] = next(iter(profiles))
        _write_config(config)
    else:
        path = _config_path()
        if path.exists():
            path.unlink()
    return True
