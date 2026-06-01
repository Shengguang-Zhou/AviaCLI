from __future__ import annotations

import json
import os
import sys
import threading
from typing import Any
from urllib import error as urlerror, parse, request


class AuthTokenManager:
    def __init__(
        self,
        *,
        api: str,
        token: str = "",
        username: str = "",
        password: str = "",
        refresh_token: str = "",
        timeout: float = 15.0,
    ) -> None:
        self.api = str(api or "").rstrip("/")
        self._token = str(token or "").strip()
        self.username = str(username or "").strip()
        self.password = str(password or "")
        self.refresh_token = str(refresh_token or "").strip()
        self.timeout = max(1.0, float(timeout or 15.0))
        self.refresh_count = 0
        self.last_refresh_error = ""
        self._lock = threading.Lock()

    @classmethod
    def from_env(cls, *, api: str, token: str = "", timeout: float = 15.0) -> AuthTokenManager:
        return cls(
            api=api,
            token=token or os.environ.get("AVIA_TOKEN", ""),
            username=os.environ.get("AVIA_EMAIL") or os.environ.get("AVIA_USERNAME", ""),
            password=os.environ.get("AVIA_PASSWORD", ""),
            refresh_token=os.environ.get("AVIA_REFRESH_TOKEN", ""),
            timeout=timeout,
        )

    def __str__(self) -> str:
        return self.token

    @property
    def token(self) -> str:
        return self._token

    @property
    def can_refresh(self) -> bool:
        return bool(self.refresh_token or (self.username and self.password))

    def export_env(self) -> dict[str, str]:
        env: dict[str, str] = {}
        if self.token:
            env["AVIA_TOKEN"] = self.token
        if self.username:
            env["AVIA_EMAIL"] = self.username
        if self.password:
            env["AVIA_PASSWORD"] = self.password
        if self.refresh_token:
            env["AVIA_REFRESH_TOKEN"] = self.refresh_token
        return env

    def ensure_token(self, *, reason: str = "startup") -> str:
        if self.token:
            return self.token
        if not self.can_refresh:
            raise RuntimeError("API token is required. Set AVIA_TOKEN or AVIA_EMAIL/AVIA_PASSWORD.")
        self.refresh(reason=reason)
        return self.token

    def refresh_after_auth_error(self, exc: object, *, label: str = "request") -> bool:
        if not is_auth_error(exc) or not self.can_refresh:
            return False
        try:
            self.refresh(reason=f"{label}_auth_expired")
            return True
        except Exception as refresh_exc:  # pragma: no cover - surfaced by caller
            self.last_refresh_error = str(refresh_exc)
            return False

    def refresh(self, *, reason: str = "manual") -> str:
        with self._lock:
            try:
                body = (
                    self._refresh_with_refresh_token()
                    if self.refresh_token
                    else self._login()
                )
            except Exception as exc:
                if not (self.refresh_token and self.username and self.password):
                    raise
                self.last_refresh_error = str(exc)
                body = self._login()
            token = str(body.get("access_token") or "").strip()
            if not token:
                raise RuntimeError("auth refresh did not return access_token")
            self._token = token
            new_refresh = str(body.get("refresh_token") or "").strip()
            if new_refresh:
                self.refresh_token = new_refresh
            self.refresh_count += 1
            self.last_refresh_error = ""
            print(f"refreshed API token for {reason}", file=sys.stderr, flush=True)
            return self._token

    def _auth_url(self, suffix: str) -> str:
        return f"{self.api.rstrip('/')}/auth/{suffix.lstrip('/')}"

    def _refresh_with_refresh_token(self) -> dict[str, Any]:
        data = json.dumps({"refresh_token": self.refresh_token}).encode("utf-8")
        req = request.Request(
            self._auth_url("refresh"),
            data=data,
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            method="POST",
        )
        return _read_json(req, timeout=self.timeout)

    def _login(self) -> dict[str, Any]:
        data = parse.urlencode({"username": self.username, "password": self.password}).encode()
        req = request.Request(
            self._auth_url("login"),
            data=data,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method="POST",
        )
        return _read_json(req, timeout=self.timeout)


def _read_json(req: request.Request, *, timeout: float) -> dict[str, Any]:
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except urlerror.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:1000]
        raise RuntimeError(f"auth request failed: HTTP {exc.code}: {detail}") from exc
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))


def is_auth_error(exc: object) -> bool:
    status = getattr(exc, "status", None) or getattr(exc, "code", None)
    try:
        if int(status or 0) in {401, 403}:
            return True
    except Exception:
        pass
    text = " ".join(
        str(getattr(exc, name, "") or "") for name in ("detail", "reason", "stderr", "stdout")
    ).lower()
    return "token_expired" in text or "unauthorized" in text or "returned error: 401" in text


def result_is_auth_failure(result: dict[str, Any]) -> bool:
    code = int(result.get("returncode") or 0)
    if code not in {22, 401, 403}:
        return False
    return is_auth_error(type("_ProbeError", (), dict(result))())


def refresh_after_auth_error(token: object, exc: object, *, label: str) -> bool:
    refresher = getattr(token, "refresh_after_auth_error", None)
    if callable(refresher):
        return bool(refresher(exc, label=label))
    return False
