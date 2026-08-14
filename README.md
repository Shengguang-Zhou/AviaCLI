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
uv run python -W error -m compileall -q -f packages/avia-cli/src/avia_cli tests
uv run pytest -p no:cacheprovider -q
uv run ruff check packages tests
uv run ruff format --check packages tests
uv build --package avia-cli
```

Development and CI use the root-enforced uv 0.8.3 toolchain on Python 3.10, 3.11, and 3.12.
The internal runner uses system Python 3.10/3.12 plus the managed-interpreter root
`/mnt/data/avia/python` for Python 3.11. CI sets both that root and `UV_PYTHON_DOWNLOADS=never`, so
a missing runtime is an observable runner error rather than an implicit network download.

Pull requests and manual internal runs use `.woodpecker/ci.yml` on Woodpecker 3.17's local backend.
The clone image is a plugin identifier, while ordinary step images are exact host executables.
The first lane hashes and executes the sole root-installed policy verifier, which binds the exact
repository/event/commit/approval, local agent, proxy, and complete host toolchain. Ordinary steps
reject clone-only NETRC credentials and use an explicit serial DAG inside each matrix workflow.
The workflow runs frozen dependency sync, owned-code warning gates, and Ruff lint/format checks on
Python 3.10, 3.11, and 3.12, then builds the sdist and wheel once on Python 3.12. Third-party
warnings remain visible.

The local backend is not a sandbox because the shared agent retains Docker authority. Exact-commit
human approval is therefore the security boundary; the root verifier is a fail-closed drift gate,
and release acceptance separately matches the PR head, Codex review, and accepted pipeline SHA.

## Boundary

This repo may contain public client code only. Do not add backend application
imports, Runtime worker code, database models, Celery tasks, GPU/model
libraries, vector store clients, or curation/quality algorithm implementations.

Dataset uploads fail before network or disk side effects when an image cannot be fully decoded,
the source tree contains a symbolic link or non-regular member such as a FIFO/socket/device, or
validation fails. Folder session identity is a
canonical UUIDv4 persisted atomically before the first POST; `--resume` replays that exact
pending request. Inspection, validation, and upload always cover the complete dataset and expose
no historical truncation flags. Resume rejects a changed file set or file identity and rehashes
every file already recorded as uploaded before issuing any network request.
Only descendants of an actual source-root `.avia/imports` directory are excluded as client state.
An exact regular file named `.avia/imports`, another `.avia` path at the root, or any nested
`.avia` path is ordinary source content and is included, then validated or rejected explicitly.
Validated files remain bound to one `O_NOFOLLOW` descriptor identity through hashing and every
retry. API-issued presigned URLs are used unchanged; there is no origin rewrite, Host override,
or silent transport fallback. PUT retries are limited to typed transport failures and exactly
HTTP 408/429/500/502/503/504 responses; invalid URLs, headers, ranges, and other contract errors
fail immediately.
Folder sessions are the sole dataset-byte upload path; the historical non-idempotent archive
command was removed. API bases, server responses, remote identities, and statuses are decoded
against one canonical contract. All untrusted JSON—HTTP, resume/cleanup state, auth config, and
COCO—uses one decoder that rejects duplicate keys at every nesting level and rejects every
non-finite decoded number, including finite-syntax overflow. Import ids are one 5..64-character
ASCII path component: `imp_` plus a non-empty
sequence of letters, digits, underscore, or hyphen; path separators, dot forms, percent encodings,
whitespace, Unicode, controls, and overlong identities are rejected. Any allowed suffix character
may appear first.
Transport concurrency is tuned from the validated storage URL, not the control-plane API host.

Object-prefix imports accept only a bare NFC POSIX object path with a trailing slash, validated
before authentication or HTTP. YOLO source imports send the validated caller class list,
Anomalib/AD automatically sends exactly `["good", "bad"]`, and COCO/ImageNet send an empty list;
`--class` is YOLO-only. Source-import success requires the control plane's complete
pre-materialization counters (`uploaded == file_count`, `image_count == streamed == 0`) rather
than a reduced response shape. Its S3 `source_version_id` remains an opaque exact string: Unicode,
whitespace, and newlines are preserved; 1..1024 UTF-8 bytes are accepted except for the exact
`null` sentinel. COCO JSON rejects duplicate fields and non-finite numbers; every
annotation requires positive finite `area` and strict `iscrowd` in `0|1`, and every split repeats
the complete category identity after canonical ordering by category id, including extension fields.
Canonical JSON identity ignores category-array and object-key order but preserves nested array
order and JSON value types. All pose categories declare
one identical `[K,3]` keypoint/skeleton schema with complete left/right pairs; skeleton edges are
unique as undirected pairs. Missing skeleton is
`[]`, `num_keypoints` may be absent, and visibility-zero coordinates remain bounded by the image.
COCO polygon
and RLE masks share one strict acceptance contract, and uncompressed RLE permits zero only in its
first run: one connected component, no holes, and pixel-exact reconstruction from one full YOLO
contour.
YOLO segment input follows AviaTraining/Ultralytics runtime semantics: official thin-bridge
multi-segment walks are valid, while rasterizable crossing topology is surfaced as a structured
warning rather than misclassified as an upload-blocking error.
YOLO `nc` is a strict class count: 10000 is the valid catalog-size ceiling, while class ids remain
bounded to `0..9999`; only an implicit plain canonical-decimal YAML scalar is accepted, so tags,
anchors, quotes, signs, padding, separators, and other coercible forms are rejected.
Anomalib uses one explicit training layout: `train/good`, `val/{good,bad}`,
`test/{good,bad}`, and `ground_truth/{val,test}/bad/<same-stem>.png`. The CLI rejects original
MVTec defect-name folders, `validation`, `_mask` filename adaptation, nested role directories,
missing roles, and missing masks before creating an upload session. Its only AD taxonomy is
`["good", "bad"]`. Inspection, validation, manifest generation, and upload share one role
inventory and report separate `image_count`, `label_count`, and `mask_count` values; Anomalib
masks are never counted as source images or labels. Source suffixes are lowercase
`.jpg | .jpeg | .png | .webp` and masks use lowercase `.png`. Decoded source encodings must be
JPEG, PNG, or WebP exactly as declared by that suffix, and masks must decode as PNG; renamed
BMP/TIFF bytes are invalid. YOLO, COCO, and ImageNet continue to preserve supported uppercase
image suffixes while validating their decoded format case-insensitively. ImageNet builds one
`(split, class)` image index before validating classes rather than rescanning the full inventory.

Signed folder responses reproduce the dataset session workspace. Before PUT, every object key
equals the session's exact `<import-prefix>/files/<relative-path>` identity; batch completion and
polling remain bound to that workspace. Signed media types use one strict lowercase ASCII
`token/token` contract with no parameters, whitespace, controls, empty token, or extra slash.
Concurrent PUTs always settle before return. Even when a periodic resume-state write fails, every
running success is recorded and a final durable save is attempted before the complete failure set
is raised; batch-complete work follows the same no-background-side-effect rule.
Import responses do not expose dataset-version identity before publication; only `succeeded`
requires the exact ten-field materialized lakeFS `version_ref`. The result id is exactly
`dsv_<import_id suffix>` and its scope-bound prefix is exactly
`dataset-manifests/<project_scope_id>/<dataset_version_id>`; the tag and exact
`<path_prefix>/manifest.json`, digest, and positive counts close the same identity.

With the local backend, `woodpeckerci/plugin-git:2.9.2` identifies the host `plugin-git` clone
binary; it is not an OCI image pin. The clone disables LFS and partial clone. Its complete shallow
fetch makes checkout and reset local, so a successfully fetched commit cannot later fail on a
second promisor-remote TLS request. This clone is the only boundary that may receive Woodpecker
NETRC variables. Before repository-owned quality commands, the root policy verifier proves the
live pipeline approval and toolchain identity through its root broker. Ordinary steps run absolute
`/usr/bin/bash`, fail on any non-empty `CI_NETRC_*`, and use the shared
`/mnt/data/avia/cache/uv`, `UV_LINK_MODE=hardlink`, and
`UV_PYTHON_DOWNLOADS=never`. The server-owned GitHub URL proxy is carried through `GIT_CONFIG_*`.
The workflow has no custom checkout path and the source-boundary gate rejects tracked Git LFS
pointer files.

## Release

Releases use PyPI Trusted Publishing through the release-only GitHub Actions workflow. Pull
request and main quality gates run only in the internal Woodpecker workflow.

1. Configure the Trusted Publisher for `avia-cli` on TestPyPI and PyPI with
   repository `Shengguang-Zhou/AviaCLI`, workflow `publish.yml`, and
   environments `testpypi` and `pypi`, respectively.
2. Run the manual workflow for TestPyPI.
3. Verify TestPyPI install.
4. Push a version tag such as `v0.1.0` to publish the package to PyPI.
