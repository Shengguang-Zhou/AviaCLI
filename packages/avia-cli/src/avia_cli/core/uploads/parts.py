from __future__ import annotations

from pathlib import Path

from avia_cli.core.uploads.transfer import (
    put_file_part as _transfer_put_file_part,
    put_file_part_with_retries as _retry_put_file_part,
)

_UPLOAD_CHUNK_SIZE = 4 * 1024 * 1024


def _put_file_part(
    *,
    upload_url: str,
    path: Path,
    offset: int,
    size: int,
    headers: dict[str, object],
) -> str:
    return _transfer_put_file_part(
        upload_url=upload_url,
        path=path,
        offset=offset,
        size=size,
        headers=headers,
        upload_chunk_size=_UPLOAD_CHUNK_SIZE,
    )


def _put_file_part_with_retries(
    *,
    upload_url: str,
    path: Path,
    offset: int,
    size: int,
    headers: dict[str, object],
    retries: int,
    base_delay_sec: float,
) -> str:
    return _retry_put_file_part(
        put_part_callback=_put_file_part,
        upload_url=upload_url,
        path=path,
        offset=offset,
        size=size,
        headers=headers,
        retries=retries,
        base_delay_sec=base_delay_sec,
    )
