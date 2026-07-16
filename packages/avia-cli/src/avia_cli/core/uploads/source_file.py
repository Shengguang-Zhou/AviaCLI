from __future__ import annotations

import errno
import os
import stat
from pathlib import Path
from types import TracebackType
from typing import BinaryIO, TypedDict


class SourceIdentity(TypedDict):
    device: int
    inode: int
    size_bytes: int
    mtime_ns: int
    ctime_ns: int


class SourceFileChangedError(RuntimeError):
    pass


def _identity_from_stat(value: os.stat_result) -> SourceIdentity:
    return {
        "device": int(value.st_dev),
        "inode": int(value.st_ino),
        "size_bytes": int(value.st_size),
        "mtime_ns": int(value.st_mtime_ns),
        "ctime_ns": int(value.st_ctime_ns),
    }


def _normalized_identity(value: SourceIdentity | dict[str, object]) -> SourceIdentity:
    required = {"device", "inode", "size_bytes", "mtime_ns", "ctime_ns"}
    if set(value) != required:
        raise ValueError(f"source identity fields must be exact: {sorted(required)}")
    if any(
        isinstance(value[key], bool) or not isinstance(value[key], int) or value[key] < 0
        for key in required
    ):
        raise ValueError("source identity values must be non-negative integers")
    identity = {key: value[key] for key in required}
    return SourceIdentity(**identity)


def capture_source_identity(path: Path) -> SourceIdentity:
    try:
        value = path.lstat()
    except OSError as exc:
        raise SourceFileChangedError(f"source file cannot be inspected: {path}: {exc}") from exc
    if stat.S_ISLNK(value.st_mode):
        raise SourceFileChangedError(f"source file is a symbolic link: {path}")
    if not stat.S_ISREG(value.st_mode):
        raise SourceFileChangedError(f"source file is not a regular file: {path}")
    return _identity_from_stat(value)


def assert_source_identity(
    path: Path,
    expected_identity: SourceIdentity | dict[str, object],
    *,
    context: str = "source file identity changed",
) -> None:
    expected = _normalized_identity(expected_identity)
    actual = capture_source_identity(path)
    if actual != expected:
        raise SourceFileChangedError(
            f"{context}: {path}: expected={dict(expected)} actual={dict(actual)}"
        )


class VerifiedSourceFile:
    def __init__(
        self,
        path: Path,
        expected_identity: SourceIdentity | dict[str, object],
    ) -> None:
        self.path = Path(path)
        self.identity = _normalized_identity(expected_identity)
        self._handle: BinaryIO | None = None

    def __enter__(self) -> VerifiedSourceFile:
        assert_source_identity(self.path, self.identity)
        flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        try:
            descriptor = os.open(self.path, flags)
        except OSError as exc:
            if exc.errno == errno.ELOOP:
                raise SourceFileChangedError(
                    f"source file became a symbolic link: {self.path}"
                ) from exc
            raise SourceFileChangedError(
                f"source file cannot be opened: {self.path}: {exc}"
            ) from exc
        try:
            actual = _identity_from_stat(os.fstat(descriptor))
            if actual != self.identity:
                raise SourceFileChangedError(
                    f"source file identity changed before open: {self.path}: "
                    f"expected={dict(self.identity)} actual={dict(actual)}"
                )
            self._handle = os.fdopen(descriptor, "rb", closefd=True)
        except BaseException:
            os.close(descriptor)
            raise
        return self

    def __exit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    @property
    def handle(self) -> BinaryIO:
        if self._handle is None:
            raise RuntimeError("verified source file is not open")
        return self._handle

    def fileno(self) -> int:
        return self.handle.fileno()

    def prepare(self, offset: int = 0) -> BinaryIO:
        self.assert_unchanged(context="source file identity changed before transfer")
        self.handle.seek(int(offset))
        return self.handle

    def assert_position(self, expected: int) -> None:
        actual = self.handle.tell()
        if actual != int(expected):
            raise SourceFileChangedError(
                f"source transfer length mismatch: {self.path}: expected_position={expected} "
                f"actual_position={actual}"
            )

    def assert_unchanged(self, *, context: str = "source file changed during transfer") -> None:
        actual_fd = _identity_from_stat(os.fstat(self.fileno()))
        if actual_fd != self.identity:
            raise SourceFileChangedError(
                f"{context}: {self.path}: expected={dict(self.identity)} "
                f"actual_fd={dict(actual_fd)}"
            )
        assert_source_identity(self.path, self.identity, context=context)


def open_verified_source(
    path: Path,
    expected_identity: SourceIdentity | dict[str, object] | None = None,
) -> VerifiedSourceFile:
    identity = expected_identity if expected_identity is not None else capture_source_identity(path)
    return VerifiedSourceFile(path, identity)
