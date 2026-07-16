# Avia Python Client

This repository contains the public Python command line client for Avia.

Dataset commands always require an explicit serialization format and task. The
only valid pairs are `yolo` with `detect|classify|segment|pose|obb`, `coco` with
`detect|segment|pose`, `imagenet` with `classify`, and `anomalib` with `ad`.
AviaCLI never infers a task or defaults to detection.

## Package

`avia-cli` is the single published distribution. It exposes the `avia` command
and contains internal modules for auth, dataset manifest scanning, signed
uploads, and import-session operations.

## Install

For users:

```bash
pipx install avia-cli
avia --help
```

AviaCLI supports Python 3.10, 3.11, and 3.12.

For one-off use:

```bash
uvx --from avia-cli avia --help
```

## Development

```bash
uv sync --all-packages --group dev
uv run pytest -W error -q
uv run ruff check packages tests
uv run ruff format --check packages tests
uv build --package avia-cli
```

Development and CI use the root-enforced uv 0.8.3 toolchain on Python 3.10, 3.11, and 3.12.
The internal runner pre-provisions those interpreters and runs with `UV_PYTHON_DOWNLOADS=never`,
so a missing runtime is an observable runner error rather than an implicit network download.

Pull requests and manual internal runs use `.woodpecker/ci.yml` on the local backend with the
native authenticated clone. The workflow uses uv 0.8.3 to run frozen dependency sync, the complete
warning-strict test suite, and Ruff lint/format checks on Python 3.10, 3.11, and 3.12, then builds
the sdist and wheel once on Python 3.12.

## Boundary

This repo may contain public client code only. Do not add backend application
imports, Runtime worker code, database models, Celery tasks, GPU/model
libraries, vector store clients, or curation/quality algorithm implementations.

Dataset uploads fail before network or disk side effects when an image cannot be fully decoded,
the source tree contains a symbolic link, or validation fails. Folder session identity is a
canonical UUIDv4 persisted atomically before the first POST; `--resume` replays that exact
pending request. Inspection, validation, and upload always cover the complete dataset and expose
no historical truncation flags. Resume rejects a changed file set or file identity and rehashes
every file already recorded as uploaded before issuing any network request.
Validated files remain bound to one `O_NOFOLLOW` descriptor identity through hashing and every
retry. API-issued presigned URLs are used unchanged; there is no origin rewrite, Host override,
or silent transport fallback. PUT retries are limited to typed transport failures and HTTP
408/429/5xx responses; invalid URLs, headers, ranges, and other contract errors fail immediately.
Folder sessions are the sole dataset-byte upload path; the historical non-idempotent archive
command was removed. API bases, server responses, remote identities, and statuses are decoded
against one canonical contract. Transport concurrency is tuned from the validated storage URL,
not the control-plane API host.

Object-prefix imports accept only a bare NFC POSIX object path with a trailing slash, validated
before authentication or HTTP. COCO polygon and RLE masks share one strict acceptance contract:
one connected component, no holes, and pixel-exact reconstruction from one full YOLO contour.
YOLO segment input follows AviaTraining/Ultralytics runtime semantics: official thin-bridge
multi-segment walks are valid, while rasterizable crossing topology is surfaced as a structured
warning rather than misclassified as an upload-blocking error.

Woodpecker uses the pinned native authenticated `woodpeckerci/plugin-git:2.9.2` clone with
`lfs: false`, the shared `/mnt/data/avia/cache/uv`, `UV_LINK_MODE=hardlink`, and
`UV_PYTHON_DOWNLOADS=never`. The workflow has no custom checkout path and the source-boundary gate
rejects tracked Git LFS pointer files.

## Release

Releases use PyPI Trusted Publishing through the release-only GitHub Actions workflow. Pull
request and main quality gates run only in the internal Woodpecker workflow.

1. Configure the Trusted Publisher for `avia-cli` on TestPyPI and PyPI with
   repository `Shengguang-Zhou/AviaCLI`, workflow `publish.yml`, and
   environments `testpypi` and `pypi`, respectively.
2. Run the manual workflow for TestPyPI.
3. Verify TestPyPI install.
4. Push a version tag such as `v0.1.0` to publish the package to PyPI.
