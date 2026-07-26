# AviaCLI Engineering Contract

This file is the living source of truth for repository-specific behavior. Update it in the
same change whenever the upload protocol, validation boundary, packaging, or CI path changes.

## Dataset upload invariants

- Dataset format and task are explicit and must match the exact public matrix documented in
  `README.md`; aliases and inferred defaults are invalid.
- Folder uploads generate a canonical lowercase UUIDv4 `idempotency_key`. A versioned
  pending-session state containing the exact request payload is atomically durable before the
  first session POST. `--resume` replays that payload and key after response loss.
- Resume state has one current schema. Historical state shapes are rejected, not migrated or
  accepted through compatibility branches. Resume requires the exact original file set and
  stored regular-file identities; every file already marked uploaded is rehashed through its
  verified descriptor and must match the persisted SHA-256 before any network request.
- Every image is fully decoded before any HTTP request or state write. YOLO
  validation also compares decoded dimensions with manifest dimensions when present.
- Anomalib masks use Pillow's non-deprecated `get_flattened_data()` API, so the package requires
  Pillow 12.1 or newer; dependency metadata must not claim support for an older Pillow ABI.
- The published package supports Python 3.10 through 3.12. The workspace, wheel metadata, NumPy
  1.x dependency, lock file, classifiers, internal CI, and release workflow must enforce that one
  range. Quality runs on all three interpreters; release artifacts are built and published once.
- uv 0.8.3 is the sole dependency resolver/cache ABI for local, Woodpecker, and release runs.
  Pin it in root `pyproject.toml`; do not add `uv.toml` or allow CI toolchain drift.
- The local Woodpecker runner must provision Python 3.10, 3.11, and 3.12 before accepting work.
  CI sets `UV_PYTHON_DOWNLOADS=never`, so a missing interpreter fails at the runner boundary
  instead of downloading an unpinned runtime during a quality gate.
- YOLO multilabel classification requires one label file per image. An existing empty label file
  is an explicit negative sample; a missing label file is a dataset error.
- YOLO segment validity follows the AviaTraining/Ultralytics consumer contract. Official
  `merge_multi_segment` thin-bridge walks, repeated seam points, and rasterizable self-crossing
  polygons are accepted. Runtime-invalid rows remain errors; crossing or non-canonical bridge
  topology is emitted as a structured `yolo_segment_topology` warning with exact path and line.
  COCO-to-YOLO conversion remains stricter because it must prove lossless single-polygon output.
- Dataset source trees must not contain symbolic links, including broken links and links to
  directories. Relative paths are unique canonical NFC POSIX paths. Only exact task media,
  metadata, and explicitly documented provenance files are allowed.
- Validation captures every regular-file identity. Hashing and every PUT retry must reuse an
  `O_NOFOLLOW` descriptor for that identity and fail on path, inode, size, or timestamp changes.
  Use the API-issued presigned URL unchanged; origin rewrites, Host overrides, and silent
  transport fallbacks are forbidden.
- Folder PUTs use one exact, case-insensitive header contract. Never retry invalid URLs, headers,
  source identities, or programmer errors. Retry only the typed transport
  error and HTTP 408/429/500/502/503/504 responses.
- A failed concurrent PUT or batch-complete request must not return while sibling operations are
  still running. Drain them, persist every completed side effect to resume state, expose all
  additional failures, then raise one structured aggregate error.
- Folder sessions are the only dataset-byte upload protocol. The non-idempotent archive path and
  its multipart implementation are removed rather than retained as historical compatibility code.
- API bases, server response fields, import identifiers, remote object identities, and import
  statuses have one canonical contract. Reject unknown fields, duplicate/missing signed entries,
  mismatches, redirects, and historical status aliases.
- `POST /projects/{project_id}/imports/{import_id}/complete` has one
  `avia.import-complete-queued/v1` response. A retry after response loss must decode the exact same
  persisted queued receipt, including non-empty `dispatch_mode` and `worker_task_id`; the client
  never weakens this contract or treats a partial replay response as success.
- Derive hashing/batching parameters locally. Probe transport RTT only against the first validated
  API-issued signed storage URL with a side-effect-free HEAD using the exact explicit proxy
  snapshot that PUT will use; never treat the control-plane API host as the storage host. A
  configured proxy failure must abort; never retry the probe by bypassing the proxy.
- Authentication refresh is allowed only for 401 or explicit `token_expired`, never 403. An
  explicit token must never gain environment password or refresh-token credentials from another
  identity. Keyring/config mutations are atomic and roll back on either side's failure.
- Never truncate inspection, validation, manifests, or uploads. Every selected dataset command
  evaluates the complete source tree.
- Invalid numeric options fail in argument parsing. Runtime code must not clamp invalid user
  values into a different valid value.
- Authentication, keyring, state, and JSON errors must surface their real cause and identifying
  path or URL. Do not silently retry with a different credential mechanism.

## Required gates

Run the same release gates as `.woodpecker/ci.yml`:

```bash
uv sync --frozen --all-packages --group dev
uv run ruff check packages tests
uv run ruff format --check packages tests
uv run python -W error -m compileall -q -f packages/avia-cli/src/avia_cli tests
uv run pytest -p no:cacheprovider -q
uv build --package avia-cli
```

The internal Woodpecker PR/manual workflow is the quality-gate source of truth. Under Woodpecker
3.14's local backend, `woodpeckerci/plugin-git:2.9.2` is a clone plugin identifier, not an OCI
image pin, while every ordinary step `image` is a host executable and must be absolute
`/usr/bin/bash`. The first `host-toolchain` step verifies the bootstrap-owned Bash and plugin-git
versions, `root:root 755` identities, and SHA256 values. Matrix combinations are separate
workflows; inside each workflow, `host-toolchain -> quality -> package` is an explicit DAG so no
two steps mutate the same workspace `.venv`. Keep the clone entry free of explicit environment
variables: only that clone boundary may receive Woodpecker-injected `CI_NETRC_*`; every ordinary
step must reject any non-empty `CI_NETRC_*` without printing its value. The server supplies a
GitHub-URL-scoped `GIT_CONFIG_*` proxy to `127.0.0.1:7897`, which does not proxy ordinary HTTP
clients. Native clone disables LFS and partial clone so checkout/reset cannot open a second
promisor-network fetch. Never restore `skip_clone` or a custom checkout script. Tracked sources must not be Git
LFS pointer files. All repositories share `UV_CACHE_DIR=/mnt/data/avia/cache/uv`, and this
same-device workspace must use `UV_LINK_MODE=hardlink`, never copy mode. The runner pre-provisions
all supported interpreters and the workflow sets `UV_PYTHON_DOWNLOADS=never`; CI must never fetch
a Python runtime implicitly. Compile-time warnings and pytest warnings owned by `avia_cli` or this
repository's tests fail the gate; third-party warnings remain visible and are never suppressed or
promoted by a global `-W error`. GitHub Actions is release-only for tags/manual trusted publishing
and runs the same frozen gates. Do not add hosted PR/main CI, server/GPU dependencies, or
deployment behavior to this public client repository.
