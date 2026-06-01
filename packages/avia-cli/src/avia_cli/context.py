from __future__ import annotations

import argparse
import os
from urllib import parse

from avia_cli.stores.keyring import CliAuthProfile, load_cli_auth_profile
from avia_sdk.auth.tokens import AuthTokenManager


def _load_saved_auth_profile() -> CliAuthProfile | None:
    try:
        return load_cli_auth_profile()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


def _env_has_refresh_credentials() -> bool:
    return bool(
        os.environ.get("AVIA_REFRESH_TOKEN")
        or (
            (os.environ.get("AVIA_EMAIL") or os.environ.get("AVIA_USERNAME"))
            and os.environ.get("AVIA_PASSWORD")
        )
    )


def api_from_args(args: argparse.Namespace) -> str:
    api = str(getattr(args, "api", "") or os.environ.get("AVIA_API_BASE", "")).strip()
    if not api:
        profile = _load_saved_auth_profile()
        if profile is not None:
            api = profile.api
    if not api:
        raise SystemExit(
            "API base is required. Pass --api, set AVIA_API_BASE, "
            "or run `avia auth login --api ...`."
        )
    return api.rstrip("/")


def token_from_args(args: argparse.Namespace, *, api: str) -> AuthTokenManager:
    token = str(getattr(args, "token", "") or os.environ.get("AVIA_TOKEN", "")).strip()
    if not token:
        profile = _load_saved_auth_profile()
        if profile is not None:
            if profile.api.rstrip("/") == str(api).rstrip("/"):
                token = profile.token
            elif not _env_has_refresh_credentials():
                raise SystemExit(
                    "Saved Avia CLI auth profile targets "
                    f"{profile.api}, but this command targets {api}. "
                    "Pass --token, set AVIA_TOKEN, or run `avia auth login` for this API."
                )
    manager = AuthTokenManager.from_env(api=api, token=token)
    try:
        manager.ensure_token(reason="cli_startup")
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    return manager


def project_path(api: str, project_id: str, suffix: str) -> str:
    return (
        f"{api.rstrip('/')}/projects/{parse.quote(str(project_id), safe='')}/"
        f"{suffix.lstrip('/')}"
    )
