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
    assert detect_storage_kind("192.168.1.9") == "lan"
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


def test_probe_rtt_seconds_exposes_target_when_connect_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_args: object, **_kwargs: object) -> object:
        raise OSError("no route to host")

    monkeypatch.setattr("avia_cli.core.uploads.autotune.socket.create_connection", boom)

    with pytest.raises(RuntimeError, match=r"192\.0\.2\.1:9.*no route to host"):
        probe_rtt_seconds("192.0.2.1", 9, "wan")


def test_probe_rtt_seconds_measures_when_connect_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeConn:
        def __enter__(self) -> _FakeConn:
            return self

        def __exit__(self, *_exc: object) -> bool:
            return False

    monkeypatch.setattr(
        "avia_cli.core.uploads.autotune.socket.create_connection",
        lambda *_a, **_k: _FakeConn(),
    )
    rtt = probe_rtt_seconds("127.0.0.1", 9000, "local")
    # a real measurement (>=0), well below the fallback
    assert rtt >= 0.0
    assert rtt < 0.0005


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
        lambda host, port, storage_kind: {
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
        upload_url="http://192.168.1.10:9000/bucket/key?X-Amz-Signature=secret",
    )
    assert args.concurrency is not None
    error_output = capsys.readouterr().err
    assert "storage_host=192.168.1.10:9000" in error_output
    assert "storage=lan" in error_output
    assert "secret" not in error_output
