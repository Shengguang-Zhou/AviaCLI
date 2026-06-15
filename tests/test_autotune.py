from __future__ import annotations

from types import SimpleNamespace

import pytest

from avia_cli.core.uploads.autotune import (
    compute_upload_params,
    detect_storage_kind,
    probe_rtt_seconds,
)
from avia_cli.core.uploads.dataset import _resolve_auto_upload_params

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


def test_compute_upload_params_unknown_kind_falls_back_to_lan() -> None:
    lan = compute_upload_params(cores=8, storage_kind="lan", probe_rtt_s=0.002)
    unknown = compute_upload_params(cores=8, storage_kind="something-else", probe_rtt_s=0.002)
    assert unknown == lan


def test_probe_rtt_seconds_falls_back_when_connect_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*_args: object, **_kwargs: object) -> object:
        raise OSError("no route to host")

    monkeypatch.setattr("avia_cli.core.uploads.autotune.socket.create_connection", boom)

    assert probe_rtt_seconds("192.0.2.1", 9, "wan") == pytest.approx(0.03)
    assert probe_rtt_seconds("192.0.2.1", 9, "local") == pytest.approx(0.0005)
    assert probe_rtt_seconds("192.0.2.1", 9, "lan") == pytest.approx(0.002)
    # unknown kind also falls back (to the wan default)
    assert probe_rtt_seconds("192.0.2.1", 9, "mystery") == pytest.approx(0.03)


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
        upload_url_origin_override="http://127.0.0.1:9000",
    )


def _auto_args() -> SimpleNamespace:
    return SimpleNamespace(
        concurrency=None,
        batch_size=None,
        hash_workers=None,
        batch_complete_concurrency=None,
        stream_flush_size=None,
        upload_url_origin_override="http://127.0.0.1:9000",
    )


@pytest.fixture(autouse=True)
def _no_network_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    # keep _resolve_auto_upload_params tests fast/offline: force the RTT
    # fallback instead of opening a real TCP socket.
    monkeypatch.setattr(
        "avia_cli.core.uploads.dataset.probe_rtt_seconds",
        lambda host, port, storage_kind: {
            "local": 0.0005,
            "lan": 0.002,
            "wan": 0.03,
        }.get(storage_kind, 0.03),
    )


def test_resolve_auto_upload_params_fills_none_values(
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = _auto_args()
    _resolve_auto_upload_params(args, api="http://127.0.0.1:6100/api/v1")
    for name in _AUTO_PARAM_NAMES:
        assert getattr(args, name) is not None
        assert isinstance(getattr(args, name), int)
    err = capsys.readouterr().err
    assert "auto upload params:" in err
    assert "storage=local" in err


def test_resolve_auto_upload_params_keeps_explicit_values(
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = _explicit_args()
    before = {name: getattr(args, name) for name in _AUTO_PARAM_NAMES}
    _resolve_auto_upload_params(args, api="http://127.0.0.1:6100/api/v1")
    after = {name: getattr(args, name) for name in _AUTO_PARAM_NAMES}
    assert after == before
    # nothing was auto-filled, so no log line should be emitted
    assert "auto upload params:" not in capsys.readouterr().err


def test_resolve_auto_upload_params_uses_api_host_when_no_override(
    capsys: pytest.CaptureFixture[str],
) -> None:
    args = _auto_args()
    args.upload_url_origin_override = ""
    _resolve_auto_upload_params(args, api="http://127.0.0.1:6100/api/v1")
    assert args.concurrency is not None
    assert "storage=local" in capsys.readouterr().err
