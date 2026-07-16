from __future__ import annotations

import os
from pathlib import Path

import pytest
import requests

from avia_cli.core.errors import (
    _UploadHTTPError,
    _UploadTransportError,
    _is_retryable_upload_error,
)
from avia_cli.core.uploads.api import _put_file_with_retries
from avia_cli.core.uploads.source_file import (
    SourceFileChangedError,
    capture_source_identity,
)
from avia_cli.core.uploads.transfer import UploadTransportRoute


class _Response:
    status_code = 200
    reason = "OK"
    text = ""


class _Session:
    def __init__(self, put) -> None:
        self.put = put
        self.trust_env = True

    def __enter__(self):
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def _route(url: str = "https://objects.example/upload") -> UploadTransportRoute:
    return UploadTransportRoute(
        upload_url=url,
        proxy_items=(("all", None), ("http", None), ("https", None)),
    )


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (_UploadTransportError("PUT"), True),
        (_UploadHTTPError(status=408, reason="timeout", detail=""), True),
        (_UploadHTTPError(status=429, reason="rate limited", detail=""), True),
        (_UploadHTTPError(status=503, reason="unavailable", detail=""), True),
        (_UploadHTTPError(status=400, reason="bad request", detail=""), False),
        (_UploadHTTPError(status=403, reason="forbidden", detail=""), False),
        (_UploadHTTPError(status=409, reason="conflict", detail=""), False),
        (ValueError("programmer error"), False),
    ],
)
def test_upload_retry_classifier_is_exact(error: BaseException, expected: bool) -> None:
    assert _is_retryable_upload_error(error) is expected


def test_folder_put_rejects_symlink_replacement_before_any_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "sample.bin"
    path.write_bytes(b"validated")
    identity = capture_source_identity(path)
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"different")
    path.unlink()
    path.symlink_to(replacement)
    request_count = 0

    def forbidden_put(*_args: object, **_kwargs: object) -> _Response:
        nonlocal request_count
        request_count += 1
        return _Response()

    monkeypatch.setattr("requests.Session", lambda: _Session(forbidden_put))

    with pytest.raises(SourceFileChangedError, match="symbolic link"):
        _put_file_with_retries(
            route=_route(),
            path=path,
            expected_identity=identity,
            headers={},
            retries=1,
            base_delay_sec=0.001,
        )

    assert request_count == 0


def test_folder_put_uses_the_explicit_frozen_route_proxy_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "sample.bin"
    path.write_bytes(b"validated")
    identity = capture_source_identity(path)
    proxy_settings = {"https": "http://proxy.example:3128"}
    observed: list[dict[str, object]] = []

    def put(*_args: object, **kwargs: object) -> _Response:
        kwargs["data"].read()  # type: ignore[union-attr]
        observed.append(kwargs)
        return _Response()

    session = _Session(put)
    monkeypatch.setattr("requests.Session", lambda: session)
    route = UploadTransportRoute(
        upload_url="https://objects.example/upload",
        proxy_items=tuple(proxy_settings.items()),
    )

    _put_file_with_retries(
        route=route,
        path=path,
        expected_identity=identity,
        headers={},
        retries=1,
        base_delay_sec=0.001,
    )

    assert len(observed) == 1
    assert observed[0]["proxies"] == proxy_settings
    assert session.trust_env is False


def test_folder_put_never_reloads_environment_after_route_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "sample.bin"
    path.write_bytes(b"validated")
    identity = capture_source_identity(path)
    frozen_proxies = {"all": "http://frozen-proxy.example:3128"}

    monkeypatch.setattr(
        "requests.sessions.get_environ_proxies",
        lambda *_args, **_kwargs: pytest.fail("PUT reloaded environment proxies"),
    )

    def send(
        _session: requests.Session,
        request: requests.PreparedRequest,
        **kwargs: object,
    ) -> requests.Response:
        assert kwargs["proxies"] == frozen_proxies
        request.body.read()
        response = requests.Response()
        response.status_code = 200
        response.reason = "OK"
        response._content = b""
        response._content_consumed = True
        return response

    monkeypatch.setattr("requests.Session.send", send)

    _put_file_with_retries(
        route=UploadTransportRoute(
            upload_url="https://objects.example/upload",
            proxy_items=tuple(frozen_proxies.items()),
        ),
        path=path,
        expected_identity=identity,
        headers={},
        retries=1,
        base_delay_sec=0.001,
    )


def test_folder_put_rejects_different_inode_before_any_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "sample.bin"
    path.write_bytes(b"same-size")
    identity = capture_source_identity(path)
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"same-size")
    replacement.replace(path)
    monkeypatch.setattr(
        "requests.Session",
        lambda: _Session(lambda *_args, **_kwargs: pytest.fail("changed inode reached PUT")),
    )

    with pytest.raises(SourceFileChangedError, match="identity changed"):
        _put_file_with_retries(
            route=_route(),
            path=path,
            expected_identity=identity,
            headers={},
            retries=1,
            base_delay_sec=0.001,
        )


def test_folder_put_detects_in_place_change_before_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "sample.bin"
    original = b"validated-payload"
    path.write_bytes(original)
    identity = capture_source_identity(path)

    def mutate_during_put(
        _url: str, *, data, headers: dict[str, str], **_kwargs: object
    ) -> _Response:
        assert headers["Content-Length"] == str(len(original))
        assert data.read() == original
        with path.open("ab") as handle:
            handle.write(b"changed")
            handle.flush()
            os.fsync(handle.fileno())
        return _Response()

    monkeypatch.setattr("requests.Session", lambda: _Session(mutate_during_put))

    with pytest.raises(SourceFileChangedError, match="changed during transfer"):
        _put_file_with_retries(
            route=_route(),
            path=path,
            expected_identity=identity,
            headers={},
            retries=1,
            base_delay_sec=0.001,
        )


def test_folder_retry_reuses_one_verified_fd_and_one_requests_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "sample.bin"
    payload = b"validated-payload"
    path.write_bytes(payload)
    identity = capture_source_identity(path)
    fds: list[int] = []
    attempts = 0

    def flaky_put(_url: str, *, data, **_kwargs: object) -> _Response:
        nonlocal attempts
        attempts += 1
        fds.append(data.fileno())
        assert data.read() == payload
        if attempts == 1:
            raise requests.exceptions.ConnectionError("connection reset")
        return _Response()

    monkeypatch.setattr("requests.Session", lambda: _Session(flaky_put))

    _put_file_with_retries(
        route=_route(),
        path=path,
        expected_identity=identity,
        headers={},
        retries=2,
        base_delay_sec=0.001,
    )

    assert attempts == 2
    assert len(set(fds)) == 1


@pytest.mark.parametrize(
    "contract_error",
    [
        requests.exceptions.InvalidURL("invalid signed URL"),
        requests.exceptions.InvalidHeader("invalid signed header"),
        requests.exceptions.MissingSchema("missing URL scheme"),
        requests.exceptions.InvalidSchema("invalid URL scheme"),
    ],
)
def test_requests_contract_errors_are_never_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contract_error: requests.exceptions.RequestException,
) -> None:
    path = tmp_path / "sample.bin"
    path.write_bytes(b"payload")
    identity = capture_source_identity(path)
    attempts = 0

    def invalid_put(*_args: object, **_kwargs: object) -> _Response:
        nonlocal attempts
        attempts += 1
        raise contract_error

    monkeypatch.setattr("requests.Session", lambda: _Session(invalid_put))
    monkeypatch.setattr(
        "avia_cli.core.uploads.timing.time.sleep",
        lambda _seconds: pytest.fail("request contract errors must not be retried"),
    )

    with pytest.raises(RuntimeError, match="folder PUT request contract is invalid"):
        _put_file_with_retries(
            route=_route(),
            path=path,
            expected_identity=identity,
            headers={},
            retries=3,
            base_delay_sec=0.001,
        )

    assert attempts == 1


@pytest.mark.parametrize(
    "headers",
    [
        {"X-Signed": "line-one\r\nX-Injected: true"},
        {"Bad Header": "value"},
        {"X-Signed": " leading-space"},
    ],
)
def test_invalid_signed_headers_fail_before_session_creation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    headers: dict[str, str],
) -> None:
    path = tmp_path / "sample.bin"
    path.write_bytes(b"payload")
    identity = capture_source_identity(path)
    monkeypatch.setattr(
        "requests.Session",
        lambda: pytest.fail("invalid signed headers reached requests.Session"),
    )
    monkeypatch.setattr(
        "avia_cli.core.uploads.timing.time.sleep",
        lambda _seconds: pytest.fail("invalid signed headers must not be retried"),
    )

    with pytest.raises(RuntimeError, match="upload required_headers"):
        _put_file_with_retries(
            route=_route(),
            path=path,
            expected_identity=identity,
            headers=headers,
            retries=3,
            base_delay_sec=0.001,
        )


@pytest.mark.parametrize("status", [301, 302, 307, 308])
def test_folder_put_never_follows_or_accepts_redirects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: int,
) -> None:
    path = tmp_path / "sample.bin"
    path.write_bytes(b"payload")
    identity = capture_source_identity(path)
    requests_seen: list[dict[str, object]] = []

    class RedirectResponse:
        status_code = status
        reason = "Redirect"
        text = ""

    def redirect_put(*_args: object, **kwargs: object) -> RedirectResponse:
        requests_seen.append(dict(kwargs))
        assert kwargs["data"].read() == b"payload"
        return RedirectResponse()

    monkeypatch.setattr("requests.Session", lambda: _Session(redirect_put))
    monkeypatch.setattr(
        "avia_cli.core.uploads.timing.time.sleep",
        lambda _seconds: pytest.fail("redirects must not be retried"),
    )

    with pytest.raises(_UploadHTTPError, match=rf"HTTP {status}"):
        _put_file_with_retries(
            route=_route(),
            path=path,
            expected_identity=identity,
            headers={},
            retries=3,
            base_delay_sec=0.001,
        )

    assert len(requests_seen) == 1
    assert requests_seen[0]["allow_redirects"] is False


@pytest.mark.parametrize(
    "headers",
    [
        {"host": "objects.invalid"},
        {"content-length": "999"},
        {"Transfer-Encoding": "chunked"},
    ],
)
def test_folder_put_contract_errors_fail_before_request_without_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    headers: dict[str, str],
) -> None:
    path = tmp_path / "sample.bin"
    path.write_bytes(b"payload")
    identity = capture_source_identity(path)
    request_count = 0

    def forbidden_put(*_args: object, **_kwargs: object) -> _Response:
        nonlocal request_count
        request_count += 1
        return _Response()

    monkeypatch.setattr("requests.Session", lambda: _Session(forbidden_put))
    monkeypatch.setattr(
        "avia_cli.core.uploads.timing.time.sleep",
        lambda _seconds: pytest.fail("contract failure must not be retried"),
    )

    with pytest.raises(RuntimeError):
        _put_file_with_retries(
            route=_route(),
            path=path,
            expected_identity=identity,
            headers=headers,
            retries=3,
            base_delay_sec=0.001,
        )

    assert request_count == 0


def test_folder_transport_has_no_direct_or_curl_fallback() -> None:
    source = (
        Path(__file__).resolve().parents[1]
        / "packages"
        / "avia-cli"
        / "src"
        / "avia_cli"
        / "core"
        / "uploads"
        / "transfer.py"
    ).read_text(encoding="utf-8")

    assert "put_file_curl" not in source
    assert "trust_env = False" in source
    assert "subprocess.run" not in source
