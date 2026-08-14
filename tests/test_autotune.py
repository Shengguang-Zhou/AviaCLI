from __future__ import annotations

from types import SimpleNamespace

import pytest

from avia_cli.core.uploads.autotune import (
    compute_upload_params,
    detect_storage_kind,
    probe_rtt_seconds,
)
from avia_cli.core.uploads.dataset import (
    _resolve_local_upload_params,
    _resolve_transport_concurrency,
)
from avia_cli.core.uploads.transfer import UploadTransportRoute

_AUTO_PARAM_NAMES = (
    "concurrency",
    "batch_size",
    "hash_workers",
    "batch_complete_concurrency",
    "stream_flush_size",
)


def test_detect_storage_kind_local_lan_wan() -> None:
    assert detect_storage_kind("127.0.0.1") == "local"
    assert detect_storage_kind("localhost") == "local"
    assert detect_storage_kind("::1") == "local"
    assert detect_storage_kind("0.0.0.0") == "local"
    assert detect_storage_kind("minio.internal.example.com") == "local"
    assert detect_storage_kind("192.168.1.13") == "lan"
    assert detect_storage_kind("10.0.0.5") == "lan"
    assert detect_storage_kind("172.16.5.4") == "lan"
    assert detect_storage_kind("172.31.255.1") == "lan"
    assert detect_storage_kind("s3.amazonaws.com") == "wan"
    assert detect_storage_kind("172.32.0.1") == "wan"


def test_compute_upload_params_cpu_floor_dominates_on_local_host() -> None:
    params = compute_upload_params(cores=32, storage_kind="local", probe_rtt_s=0.0005)
    assert params["concurrency"] == 64
    assert params["batch_complete_concurrency"] == 8
    assert params["hash_workers"] == 16
    assert params["batch_size"] == 512
    assert params["stream_flush_size"] == 512


def test_compute_upload_params_wan_high_rtt_clamped_low_cores() -> None:
    params = compute_upload_params(cores=4, storage_kind="wan", probe_rtt_s=0.1)
    assert params["concurrency"] >= 8
    assert params["concurrency"] <= min(256, 16 * 4)
    # 4 cores -> upper bound is 64
    assert params["concurrency"] <= 64


def test_compute_upload_params_floor_is_eight() -> None:
    # single core, fast WAN link: cpu_floor=2 and conc_lat~1 are below the
    # clamp floor of 8, so the floor (not the drivers) sets concurrency.
    params = compute_upload_params(cores=1, storage_kind="wan", probe_rtt_s=0.0005)
    assert params["concurrency"] == 8


def test_compute_upload_params_upper_bound_scales_with_cores() -> None:
    # tiny cores cap concurrency at 16*cores even when drivers want more
    params = compute_upload_params(cores=2, storage_kind="wan", probe_rtt_s=1.0)
    assert params["concurrency"] == 16 * 2  # capped at 32


def test_compute_upload_params_rejects_unknown_storage_kind() -> None:
    with pytest.raises(ValueError, match="unsupported storage kind"):
        compute_upload_params(cores=8, storage_kind="something-else", probe_rtt_s=0.002)


def test_probe_rtt_seconds_exposes_target_when_request_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import requests

    class _FailingSession:
        def __enter__(self) -> _FailingSession:
            return self

        def __exit__(self, *_exc: object) -> bool:
            return False

        def head(self, *_args: object, **_kwargs: object) -> object:
            raise requests.ConnectionError("no route to host")

    monkeypatch.setattr("requests.Session", _FailingSession)

    route = UploadTransportRoute(
        upload_url="https://192.0.2.1/upload?signature=secret",
        proxy_items=(),
    )
    with pytest.raises(RuntimeError, match=r"192\.0\.2\.1:443.*ConnectionError"):
        probe_rtt_seconds(route, "wan")


def test_probe_rtt_seconds_measures_when_connect_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    proxy_settings = {"https": "http://proxy.example:3128"}
    calls: list[tuple[str, dict[str, object]]] = []

    class _FakeResponse:
        status_code = 403

        def __enter__(self) -> _FakeResponse:
            return self

        def __exit__(self, *_exc: object) -> bool:
            return False

    class _FakeSession:
        def __init__(self) -> None:
            self.trust_env = True

        def __enter__(self) -> _FakeSession:
            return self

        def __exit__(self, *_exc: object) -> bool:
            return False

        def head(self, url: str, **kwargs: object) -> _FakeResponse:
            calls.append((url, kwargs))
            return _FakeResponse()

    session = _FakeSession()
    monkeypatch.setattr("requests.Session", lambda: session)

    url = "https://objects.example/upload?X-Amz-Signature=secret"
    route = UploadTransportRoute(upload_url=url, proxy_items=tuple(proxy_settings.items()))
    rtt = probe_rtt_seconds(route, "wan")

    assert rtt >= 0.0
    assert len(calls) == 3
    assert all(call_url == url for call_url, _kwargs in calls)
    assert all(kwargs["proxies"] == proxy_settings for _url, kwargs in calls)
    assert all(kwargs["allow_redirects"] is False for _url, kwargs in calls)
    assert session.trust_env is False


def test_probe_rtt_never_reloads_environment_after_route_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import requests

    frozen_proxies = {"all": "http://frozen-proxy.example:3128"}
    monkeypatch.setattr(
        "requests.sessions.get_environ_proxies",
        lambda *_args, **_kwargs: pytest.fail("RTT probe reloaded environment proxies"),
    )

    def send(
        _session: requests.Session,
        _request: requests.PreparedRequest,
        **kwargs: object,
    ) -> requests.Response:
        assert kwargs["proxies"] == frozen_proxies
        response = requests.Response()
        response.status_code = 403
        response.reason = "Forbidden"
        response._content = b""
        response._content_consumed = True
        return response

    monkeypatch.setattr("requests.Session.send", send)
    route = UploadTransportRoute(
        upload_url="https://objects.example/upload?signature=secret",
        proxy_items=tuple(frozen_proxies.items()),
    )

    assert probe_rtt_seconds(route, "wan") >= 0.0


def test_probe_rtt_seconds_rejects_proxy_authentication_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ProxyAuthResponse:
        status_code = 407

        def __enter__(self) -> _ProxyAuthResponse:
            return self

        def __exit__(self, *_exc: object) -> bool:
            return False

    class _ProxySession:
        def __enter__(self) -> _ProxySession:
            return self

        def __exit__(self, *_exc: object) -> bool:
            return False

        def head(self, *_args: object, **_kwargs: object) -> _ProxyAuthResponse:
            return _ProxyAuthResponse()

    monkeypatch.setattr("requests.Session", _ProxySession)

    route = UploadTransportRoute(
        upload_url="https://objects.example/upload?signature=secret",
        proxy_items=(("https", "http://proxy.example:3128"),),
    )
    with pytest.raises(RuntimeError, match=r"proxy authentication failed.*407"):
        probe_rtt_seconds(route, "wan")


def _explicit_args() -> SimpleNamespace:
    return SimpleNamespace(
        concurrency=7,
        batch_size=123,
        hash_workers=3,
        batch_complete_concurrency=5,
        stream_flush_size=9,
    )


def _auto_args() -> SimpleNamespace:
    return SimpleNamespace(
        concurrency=None,
        batch_size=None,
        hash_workers=None,
        batch_complete_concurrency=None,
        stream_flush_size=None,
    )


@pytest.fixture(autouse=True)
def _no_network_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "avia_cli.core.uploads.dataset.probe_rtt_seconds",
        lambda _route, storage_kind: {
            "local": 0.0005,
            "lan": 0.002,
            "wan": 0.03,
        }.get(storage_kind, 0.03),
    )


def test_resolve_local_upload_params_fills_only_cpu_values(
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = _auto_args()
    _resolve_local_upload_params(args)
    assert args.concurrency is None
    for name in _AUTO_PARAM_NAMES[1:]:
        assert getattr(args, name) is not None
        assert isinstance(getattr(args, name), int)
    err = capsys.readouterr().err
    assert "auto local upload params:" in err


def test_resolve_auto_upload_params_keeps_explicit_values(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _explicit_args()
    monkeypatch.setattr(
        "avia_cli.core.uploads.dataset.probe_rtt_seconds",
        lambda *_args: pytest.fail("explicit upload params must not trigger an RTT probe"),
    )
    before = {name: getattr(args, name) for name in _AUTO_PARAM_NAMES}
    _resolve_local_upload_params(args)
    after = {name: getattr(args, name) for name in _AUTO_PARAM_NAMES}
    assert after == before
    # nothing was auto-filled, so no log line should be emitted
    assert "auto local upload params:" not in capsys.readouterr().err


def test_resolve_transport_concurrency_uses_signed_storage_host(
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = _auto_args()
    _resolve_transport_concurrency(
        args,
        route=UploadTransportRoute(
            upload_url="http://192.168.1.10:9000/bucket/key?X-Amz-Signature=secret",
            proxy_items=(),
        ),
    )
    assert args.concurrency is not None
    error_output = capsys.readouterr().err
    assert "storage_host=192.168.1.10:9000" in error_output
    assert "storage=lan" in error_output
    assert "secret" not in error_output
