from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace
from urllib import request
from urllib import error as urlerror

import pytest

from avia_cli.commands.auth import _post_public_json
from avia_cli.commands.auth import _read_json as read_auth_json
from avia_cli.commands.auth import _token_from_login_args, validate_cli_token
from avia_cli.core.auth.tokens import AuthTokenManager
from avia_cli.core.auth.tokens import _read_json as read_token_json
from avia_cli.core.errors import _AviaHTTPError
from avia_cli.core.http import no_redirect
from avia_cli.core.http.form import _request_form_json
from avia_cli.core.uploads.api import _request_json, _request_json_with_retries
from avia_cli.stores.keyring import (
    clear_cli_auth_profile,
    load_cli_auth_profile,
    save_cli_auth_profile,
)


def test_token_stdin_rejects_empty_input_without_starting_device_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(" \n"))

    with pytest.raises(RuntimeError, match="--token-stdin received no token"):
        _token_from_login_args(SimpleNamespace(token="", token_stdin=True))


def test_refresh_failure_exposes_original_reason_once_without_login_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manager = AuthTokenManager(
        api="https://avia.example/api/v1",
        token="expired",
        refresh_token="revoked",
        username="user@example.com",
        password="secret",
    )
    calls = {"refresh": 0, "login": 0}

    def fail_refresh() -> dict[str, object]:
        calls["refresh"] += 1
        raise RuntimeError("refresh token was revoked")

    def forbidden_login() -> dict[str, object]:
        calls["login"] += 1
        raise AssertionError("refresh failure must not silently fall back to password login")

    monkeypatch.setattr(manager, "_refresh_with_refresh_token", fail_refresh)
    monkeypatch.setattr(manager, "_login", forbidden_login)
    auth_error = SimpleNamespace(status=401, detail="token_expired")

    with pytest.raises(RuntimeError, match="refresh token was revoked"):
        manager.refresh_after_auth_error(auth_error, label="upload")

    assert calls == {"refresh": 1, "login": 0}
    assert manager.refresh_count == 0
    assert manager.last_refresh_error == "refresh token was revoked"


def test_logout_fails_when_keyring_deletion_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FailingKeyring:
        def get_password(self, _service: str, _username: str) -> str:
            return "token"

        def set_password(self, _service: str, _username: str, _token: str) -> None:
            pass

        def delete_password(self, _service: str, _username: str) -> None:
            raise RuntimeError("keyring is locked")

    monkeypatch.setenv("AVIA_CLI_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr(
        "avia_cli.stores.keyring._read_config",
        lambda: {
            "active_profile": "default",
            "profiles": {
                "default": {
                    "api": "https://avia.example/api/v1",
                    "key_prefix": "token",
                    "keyring_username": "default:https://avia.example/api/v1",
                    "role": "editor",
                    "saved_at": "2026-07-15T00:00:00+00:00",
                    "user_id": "usr_123",
                    "workspace_id": "ws_123",
                }
            },
        },
    )
    monkeypatch.setattr("avia_cli.stores.keyring._load_keyring", lambda: FailingKeyring())

    with pytest.raises(RuntimeError, match="keyring is locked"):
        clear_cli_auth_profile()


def test_profile_save_rejects_incomplete_principal_before_keyring_write() -> None:
    with pytest.raises(RuntimeError, match="user_id must be a canonical non-empty string"):
        save_cli_auth_profile(
            api="https://avia.example/api/v1",
            token="secret",
            workspace_id="ws_123",
            user_id="",
            role="editor",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("keyring_username", "historical-default", "keyring identity"),
        ("saved_at", "2026-07-15T00:00:00", "saved_at timezone"),
        ("role", "", "role"),
    ],
)
def test_profile_load_rejects_noncanonical_config_instead_of_deriving_values(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
    message: str,
) -> None:
    monkeypatch.setenv("AVIA_CLI_CONFIG_DIR", str(tmp_path))
    metadata = {
        "api": "https://avia.example/api/v1",
        "key_prefix": "secret",
        "keyring_username": "default:https://avia.example/api/v1",
        "role": "editor",
        "saved_at": "2026-07-15T00:00:00+00:00",
        "user_id": "usr_123",
        "workspace_id": "ws_123",
    }
    metadata[field] = value
    (tmp_path / "config.json").write_text(
        json.dumps({"active_profile": "default", "profiles": {"default": metadata}}),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match=message):
        load_cli_auth_profile()


def test_token_validation_requires_complete_current_principal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "avia_cli.commands.auth._read_json",
        lambda *_args, **_kwargs: {
            "principal": {"workspace_id": "ws_123", "user_id": "", "role": "editor"}
        },
    )

    with pytest.raises(RuntimeError, match="invalid user_id"):
        validate_cli_token(api="https://avia.example/api/v1", token="token")


def test_profile_save_rolls_back_keyring_when_config_commit_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class RecordingKeyring:
        def __init__(self) -> None:
            self.value: str | None = None

        def get_password(self, _service: str, _username: str) -> str | None:
            return self.value

        def set_password(self, _service: str, _username: str, token: str) -> None:
            self.value = token

        def delete_password(self, _service: str, _username: str) -> None:
            self.value = None

    keyring = RecordingKeyring()
    monkeypatch.setenv("AVIA_CLI_CONFIG_DIR", str(tmp_path))
    monkeypatch.setattr("avia_cli.stores.keyring._load_keyring", lambda: keyring)
    monkeypatch.setattr(
        "avia_cli.stores.keyring._write_config",
        lambda _config: (_ for _ in ()).throw(OSError("disk full")),
    )

    with pytest.raises(OSError, match="disk full"):
        save_cli_auth_profile(
            api="https://avia.example/api/v1",
            token="secret",
            workspace_id="ws_123",
            user_id="usr_123",
            role="editor",
        )

    assert keyring.value is None


class _Response:
    status = 200

    def __init__(self, raw: bytes) -> None:
        self.raw = raw

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return self.raw


@pytest.mark.parametrize("reader", ["auth", "api"])
def test_invalid_http_json_is_a_structured_error_with_url(
    monkeypatch: pytest.MonkeyPatch, reader: str
) -> None:
    monkeypatch.setattr(
        "avia_cli.core.http.no_redirect.open_no_redirect",
        lambda *_args, **_kwargs: _Response(b"{"),
    )
    url = "https://avia.example/api/v1/probe"

    with pytest.raises(RuntimeError) as captured:
        if reader == "auth":
            read_auth_json(request.Request(url), timeout=1)
        else:
            _request_json(method="GET", url=url, token="token")

    payload = json.loads(str(captured.value))
    assert payload["code"] == "invalid_json_response"
    assert payload["url"] == url
    assert payload["line"] == 1
    assert payload["column"] == 2


def test_all_control_and_auth_transports_reject_redirects_without_following_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = "https://avia.example/api/v1/source"
    target = "https://attacker.example/token-capture"
    requests_seen: list[request.Request] = []

    def redirect(req: request.Request, **_kwargs: object):
        requests_seen.append(req)
        raise urlerror.HTTPError(
            req.full_url,
            302,
            "Found",
            {"Location": target},
            io.BytesIO(b"redirect"),
        )

    monkeypatch.setattr(no_redirect._OPENER, "open", redirect)
    calls = [
        lambda: _request_json(method="GET", url=source, token="control-token"),
        lambda: _request_json(
            method="POST", url=source, token="control-token", payload={"value": 1}
        ),
        lambda: _request_form_json(
            method="POST",
            url=source,
            token="form-token",
            fields={"value": 1},
        ),
        lambda: read_token_json(
            request.Request(source, headers={"Authorization": "Bearer auth-token"}),
            timeout=1,
        ),
        lambda: read_auth_json(
            request.Request(source, headers={"Authorization": "Bearer login-token"}),
            timeout=1,
        ),
        lambda: _post_public_json(
            api="https://avia.example/api/v1",
            suffix="source",
            payload={"value": 1},
            timeout=1,
        ),
    ]

    for call in calls:
        with pytest.raises(no_redirect.UnexpectedHTTPRedirect) as captured:
            call()
        payload = json.loads(str(captured.value))
        assert payload == {
            "code": "unexpected_http_redirect",
            "location": target,
            "method": requests_seen[-1].get_method(),
            "status": 302,
            "url": requests_seen[-1].full_url,
        }

    assert len(requests_seen) == len(calls)
    assert all(item.full_url != target for item in requests_seen)


def test_request_retry_layer_does_not_refresh_twice_after_repeated_401(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Token:
        def __init__(self) -> None:
            self.refresh_count = 0

        def __str__(self) -> str:
            return "expired"

        def refresh_after_auth_error(self, _exc: object, *, label: str) -> bool:
            assert label == "GET"
            self.refresh_count += 1
            return True

    def unauthorized(req: request.Request, **_kwargs: object) -> _Response:
        raise urlerror.HTTPError(
            req.full_url,
            401,
            "Unauthorized",
            {},
            io.BytesIO(b'{"error":{"message":"expired"}}'),
        )

    token = Token()
    monkeypatch.setattr("avia_cli.core.http.no_redirect.open_no_redirect", unauthorized)

    with pytest.raises(_AviaHTTPError, match="HTTP 401"):
        _request_json_with_retries(
            method="GET",
            url="https://avia.example/api/v1/probe",
            token=token,
            retries=3,
        )

    assert token.refresh_count == 1


def test_403_never_refreshes_or_switches_credentials() -> None:
    manager = AuthTokenManager(
        api="https://avia.example/api/v1",
        token="valid-but-forbidden",
        refresh_token="refresh",
    )

    assert manager.refresh_after_auth_error(SimpleNamespace(status=403), label="upload") is False
    assert manager.refresh_count == 0


def test_explicit_token_does_not_attach_environment_password_for_later_identity_switch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AVIA_EMAIL", "other@example.com")
    monkeypatch.setenv("AVIA_PASSWORD", "other-secret")
    monkeypatch.setenv("AVIA_REFRESH_TOKEN", "other-identity-refresh-token")

    manager = AuthTokenManager.from_env(api="https://avia.example/api/v1", token="explicit-token")

    assert manager.token == "explicit-token"
    assert manager.username == ""
    assert manager.password == ""
    assert manager.refresh_token == ""
    assert manager.can_refresh is False
