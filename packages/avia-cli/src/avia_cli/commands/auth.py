from __future__ import annotations

import getpass
import json
import sys
import time
import webbrowser
from typing import Any
from urllib import error as urlerror, request

from avia_cli.core.api_base import canonical_api_base
from avia_cli.core.errors import decode_json_response
from avia_cli.core.http import no_redirect
from avia_cli.stores.keyring import (
    clear_cli_auth_profile,
    load_cli_auth_profile,
    save_cli_auth_profile,
)


def _auth_me_url(api: str) -> str:
    return f"{str(api).rstrip('/')}/auth/me"


def _api_path(api: str, suffix: str) -> str:
    return f"{str(api).rstrip('/')}/{suffix.lstrip('/')}"


def _read_json(req: request.Request, *, timeout: float) -> dict[str, Any]:
    try:
        with no_redirect.open_no_redirect(req, timeout=timeout) as resp:
            raw = resp.read()
    except urlerror.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:1000]
        raise RuntimeError(f"auth validation failed: HTTP {exc.code}: {detail}") from exc
    if not raw:
        return {}
    return decode_json_response(raw, url=req.full_url)


def _post_public_json(
    *, api: str, suffix: str, payload: dict[str, Any], timeout: float
) -> tuple[int, dict[str, Any]]:
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(
        _api_path(api, suffix),
        data=data,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with no_redirect.open_no_redirect(req, timeout=timeout) as resp:
            raw = resp.read()
            status = int(getattr(resp, "status", 200) or 200)
    except urlerror.HTTPError as exc:
        raw = exc.read()
        status = int(exc.code)
    body = decode_json_response(raw, url=req.full_url) if raw else {}
    return status, dict(body or {})


def validate_cli_token(*, api: str, token: str, timeout: float = 15.0) -> dict[str, Any]:
    req = request.Request(
        _auth_me_url(api),
        headers={"Accept": "application/json", "Authorization": f"Bearer {token}"},
        method="GET",
    )
    payload = _read_json(req, timeout=timeout)
    raw_principal = payload.get("principal")
    if not isinstance(raw_principal, dict):
        raise RuntimeError("auth validation did not return a principal object")
    for field in ("workspace_id", "user_id", "role"):
        value = raw_principal.get(field)
        if not isinstance(value, str) or not value or value != value.strip():
            raise RuntimeError(f"auth validation principal has invalid {field}")
    return payload


def _token_from_login_args(args) -> str:
    token = str(getattr(args, "token", "") or "").strip()
    if token:
        return token
    if bool(getattr(args, "token_stdin", False)):
        token = sys.stdin.read().strip()
        if not token:
            raise RuntimeError("--token-stdin received no token")
        return token
    return getpass.getpass("Avia API key: ").strip()


def _manual_token_requested(args) -> bool:
    return bool(str(getattr(args, "token", "") or "").strip()) or bool(
        getattr(args, "token_stdin", False)
    )


def _device_login(args, *, api: str) -> tuple[str, dict[str, Any]]:
    status, started = _post_public_json(
        api=api,
        suffix="/cli/auth/device/start",
        payload={"client_name": "Avia CLI"},
        timeout=15.0,
    )
    if status >= 400:
        raise RuntimeError(json.dumps(started, ensure_ascii=False))

    device_code = str(started.get("device_code") or "").strip()
    user_code = str(started.get("user_code") or "").strip()
    verification_url = str(
        started.get("verification_uri_complete") or started.get("verification_uri") or ""
    ).strip()
    if not device_code or not user_code or not verification_url:
        raise RuntimeError("CLI auth device flow did not return a login URL")

    if bool(getattr(args, "no_browser", False)):
        print("Open this URL to approve Avia CLI login:")
    else:
        webbrowser.open(verification_url)
        print("Opening browser for Avia CLI login:")
    print(f"  {verification_url}")
    print(f"Code: {user_code}")

    timeout_sec = int(getattr(args, "device_timeout", 600))
    deadline = time.monotonic() + timeout_sec
    interval = getattr(args, "poll_interval", None)
    if interval is None:
        interval = float(started.get("interval") or 2)
    poll_interval = float(interval)
    if poll_interval <= 0:
        raise RuntimeError("device authorization poll interval must be greater than zero")

    while time.monotonic() < deadline:
        if poll_interval:
            time.sleep(poll_interval)
        status, polled = _post_public_json(
            api=api,
            suffix="/cli/auth/device/token",
            payload={"device_code": device_code},
            timeout=15.0,
        )
        if status == 200 and str(polled.get("access_token") or "").strip():
            token = str(polled["access_token"]).strip()
            return token, polled
        error = str(polled.get("error") or "").strip()
        if error == "authorization_pending":
            continue
        description = str(polled.get("error_description") or error or "login failed")
        raise RuntimeError(f"Avia CLI login failed: {description}")

    raise RuntimeError("Avia CLI login timed out before browser approval")


def _print_status(profile, payload: dict[str, Any], *, as_json: bool) -> None:
    principal = dict(payload["principal"])
    body = {
        "api": profile.api,
        "profile": profile.profile,
        "workspace_id": principal["workspace_id"],
        "user_id": principal["user_id"],
        "role": principal["role"],
        "key_prefix": profile.key_prefix,
    }
    if as_json:
        print(json.dumps(body, ensure_ascii=False, indent=2))
        return
    print(
        "Logged in"
        f" profile={body['profile']}"
        f" api={body['api']}"
        f" workspace={body['workspace_id']}"
        f" role={body['role']}"
        f" key={body['key_prefix']}..."
    )


def handle_auth_command(args) -> int:
    auth_command = str(getattr(args, "auth_command", "") or "")
    if auth_command == "login":
        try:
            api = canonical_api_base(getattr(args, "api", ""))
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        token = _token_from_login_args(args) if _manual_token_requested(args) else ""
        if not token:
            token, _device_payload = _device_login(args, api=api)
        payload = validate_cli_token(api=api, token=token)
        principal = dict(payload.get("principal") or {})
        profile = save_cli_auth_profile(
            api=api,
            token=token,
            workspace_id=str(principal.get("workspace_id") or ""),
            user_id=str(principal.get("user_id") or ""),
            role=str(principal.get("role") or ""),
        )
        print(
            f"Logged in to {profile.api} "
            f"(workspace={profile.workspace_id}, role={profile.role}, key={profile.key_prefix}...)"
        )
        return 0
    if auth_command == "status":
        profile = load_cli_auth_profile()
        if profile is None:
            raise SystemExit("No Avia CLI auth profile. Run `avia auth login --api ...`.")
        payload = validate_cli_token(api=profile.api, token=profile.token)
        _print_status(profile, payload, as_json=bool(getattr(args, "json", False)))
        return 0
    if auth_command == "logout":
        removed = clear_cli_auth_profile()
        print("Logged out" if removed else "No Avia CLI auth profile found")
        return 0
    raise SystemExit(f"unsupported auth command: {auth_command}")
