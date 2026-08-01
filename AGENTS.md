# AviaCLI Engineering Contract

This file is the living source of truth for repository-specific behavior. Update it in the
same change whenever the upload protocol, validation boundary, packaging, or CI path changes.

## Dataset upload invariants

- Dataset format and task are explicit and must match the exact public matrix documented in
  `README.md`; aliases and inferred defaults are invalid.
- Object-prefix source imports use one format-bound class contract: YOLO sends the caller's
  validated class list, Anomalib/AD automatically sends exactly `["good", "bad"]`, and
  COCO/ImageNet send an empty list. `--class` is a YOLO-only option. The accepted source-import
  progress is the control plane's exact pre-materialization shape, including
  `uploaded == file_count`, `image_count == 0`, and `streamed == 0`; missing or extra counters
  are protocol errors. Its S3 `source_version_id` is one opaque exact string: preserve Unicode,
  whitespace, and newlines byte-for-byte; require 1..1024 UTF-8 bytes and reject the exact `null`
  version sentinel rather than padded variants. Never trim, normalize, coerce, or reuse
  bucket/ETag validators for it.
- Folder uploads generate a canonical lowercase UUIDv4 `idempotency_key`. A versioned
  pending-session state containing the exact request payload is atomically durable before the
  first session POST. `--resume` replays that payload and key after response loss.
- Resume state has one current schema. Historical state shapes are rejected, not migrated or
  accepted through compatibility branches. Resume requires the exact original file set and
  stored regular-file identities; every file already marked uploaded is rehashed through its
  verified descriptor and must match the persisted SHA-256 before any network request. HTTP,
  resume/cleanup state, auth config, and COCO documents share one strict JSON decoder that rejects
  duplicate object keys recursively and rejects every non-finite decoded number, including
  `NaN`/`Infinity` and finite-syntax overflow; parallel permissive decoders are
  forbidden.
- Every image is fully decoded before any HTTP request or state write. YOLO
  validation also compares decoded dimensions with manifest dimensions when present. The decoded
  Pillow format must exactly match the case-normalized suffix: JPG/JPEG is `JPEG`, PNG is `PNG`,
  WebP is `WEBP`, BMP is `BMP`, and TIF/TIFF is `TIFF`; renamed encodings are invalid.
  YOLO, COCO, and ImageNet preserve supported uppercase image suffixes. Anomalib alone requires
  lowercase source suffixes and lowercase `.png` masks.
- One format-aware role inventory owns inspection, validation, manifest, and upload counts.
  Public JSON exposes `image_count`, `label_count`, and `mask_count`; Anomalib
  `ground_truth/**` members count only as masks and never inflate image or label counts. Inspect
  returns the real class catalog: COCO reuses the strict annotation-category parser and ImageNet
  reuses the same role-directory/inventory index as validation; neither fabricates an empty
  catalog or performs a second full source scan.
- Anomalib masks use Pillow's non-deprecated `get_flattened_data()` API, so the package requires
  Pillow 12.1 or newer; dependency metadata must not claim support for an older Pillow ABI.
- Anomalib has one trainable folder contract: `train/good`, both
  `val/{good,bad}` and `test/{good,bad}`, plus exact same-stem lowercase-PNG masks at
  `ground_truth/{val,test}/bad`. Original MVTec defect-name directories, `validation`, `_mask`
  suffixes, alternate split derivation, and nested role directories are invalid rather than
  adapted. Root provenance files match the importer exactly: README, LICENSE, and
  `source_records.json` only. Its source members decode only as JPEG, PNG, or WebP according to
  their suffix, and every mask decodes as PNG; renaming BMP/TIFF bytes to an allowed suffix is
  invalid. Validation, inspection, and folder-session payloads expose only `["good", "bad"]`.
- ImageNet validation builds the complete `(split, class)` image index once. Per-class rescans of
  the full image inventory are forbidden because they make validation
  `O(classes × images)` instead of `O(classes + images)`.
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
- YOLO pose accepts only `kpt_shape=[K,3]` with `1 <= K <= 2048`, matching the control plane and
  AviaTraining. COCO uses strict JSON with duplicate-field and non-finite-number rejection;
  every annotation requires positive finite `area` and strict `iscrowd` in `0|1`. Every split
  reproduces the complete categories identity after canonical ordering by category id, including
  extension fields. Category identity is canonical JSON: category-array order and object-key order
  are irrelevant, while nested array order and JSON value types are exact. All COCO pose categories
  share one exact keypoint/skeleton schema with complete
  left/right counterparts; an omitted skeleton is `[]`, `num_keypoints` is optional, and
  visibility-zero coordinates remain valid only inside the image boundary. Skeleton edges are
  unique as undirected keypoint pairs, so exact and reversed duplicates are both invalid.
  Uncompressed COCO RLE covers the exact image area and permits zero only as its first run.
- Dataset source trees must not contain symbolic links, including broken links and links to
  directories. FIFOs, sockets, devices, and every other non-regular member are rejected at the
  shared manifest boundary. Relative paths are unique canonical NFC POSIX paths. Only exact task
  media, metadata, and explicitly documented provenance files are allowed. The sole excluded
  client state is the descendants of a source-root `.avia/imports` directory plus its directory
  containers. An exact regular file named `.avia/imports`, another root `.avia` member, or any
  nested `.avia` path is dataset content and must enter the manifest and fail explicit validation
  rather than disappear.
- Validation captures every regular-file identity. Hashing and every PUT retry must reuse an
  `O_NOFOLLOW` descriptor for that identity and fail on path, inode, size, or timestamp changes.
  Use the API-issued presigned URL unchanged; origin rewrites, Host overrides, and silent
  transport fallbacks are forbidden.
- Folder PUTs use one exact, case-insensitive header contract. Never retry invalid URLs, headers,
  source identities, or programmer errors. Retry only the typed transport
  error and HTTP 408/429/500/502/503/504 responses.
- Folder signing requests contain only `relative_path`, `size_bytes`, and lowercase SHA-256.
  Content type is server-owned canonical metadata: the client validates the signed response and
  exact `Content-Type` header, persists only that server value after a successful PUT, and reuses
  it for batch completion. Browser/host MIME guesses, dimensions, and completion metadata must
  never be echoed into the signing request or substituted during resume. The one media-type
  parser accepts exactly lowercase ASCII `token/token` without whitespace, parameters, controls,
  empty sides, or additional slashes. Every signed response is bound to the persisted session
  workspace, and before PUT its object key must equal the session's exact
  `<import-prefix>/files/<relative-path>`. Batch-complete and poll responses reproduce that same
  workspace; persisted uploaded state revalidates the exact object key before resume.
- A failed concurrent PUT or batch-complete request must not return while sibling operations are
  still running. A periodic state-write failure cannot bypass executor shutdown or future
  settlement: drain every submitted future, record and make one final durable save of every
  completed side effect, expose all PUT/state/completion failures, then raise one structured
  aggregate error. Timing is observed immediately after its remote call returns, and progress may
  be emitted between bounded periodic state flushes; either telemetry failure is held while the
  completed side effect is attached to in-memory resume state and a final durable save is attempted.
  Only then is the telemetry failure reported, together with any state failure. It must never turn a
  completed PUT, batch completion, or import finalize into an unrecorded side effect.
- Folder sessions are the only dataset-byte upload protocol.
- API bases, server response fields, import identifiers, remote object identities, and import
  statuses have one canonical contract. Reject unknown fields, duplicate/missing signed entries,
  mismatches, redirects, and historical status aliases. An import id is one 5..64-character ASCII
  path component: `imp_` plus a non-empty sequence of letters, digits, underscore, or hyphen.
  Separators, dot forms, percent encodings, whitespace, Unicode, control characters, and overlong
  values are invalid identities; any of the four allowed suffix characters may be the first suffix
  character.
- Dataset-session and completion responses decode one exact nested `dataset_manifest_ref` plus
  `read_lease` model. Ref id, format, counts, canonical `project_assets/{workspace_id}/...`
  owner, MinIO manifest path/import prefix, outer object key, lease id, and lease target are bound
  to the same request and import. The exact dataset-session response is durable resume-state
  identity, and completion must reproduce its workspace, manifest ref, object key, and lease;
  object-key components must be NFC, exactly trimmed, and free of every Unicode `Cc`; loose,
  foreign, malformed, or cross-phase replacement objects are contract failures.
- A succeeded import owns exactly `dsv_<import_id suffix>`. Its materialized prefix is exactly
  `dataset-manifests/<project_scope_id>/<dataset_version_id>`, where the scope comes from the
  canonical session/list manifest owner, and its manifest is exactly `<path_prefix>/manifest.json`.
  A self-consistent response for another import or scope is an identity violation.
- Class catalogs use one validator across source imports, every dataset-format validation result,
  YOLO metadata/declared classes, folder upload, and resume state. Every name is non-empty,
  NFC-normalized, exactly trimmed, free of every Unicode `Cc` control (including DEL and C1), and unique.
  A name is at most 200 Unicode code points, a catalog contains at most 10,000 classes, and every
  canonical class id is in `0..9999`; these limits are identical across CLI, control plane,
  Training manifests, and Vision label-space refs.
  Callers never normalize or silently drop bad names, and invalid derived catalogs fail before
  API resolution or authentication. Indexed catalogs accept only a real non-negative integer or
  the canonical decimal string `0|[1-9][0-9]*`, require unique contiguous zero-based indices, and
  reject booleans, floats, signs, padding, and leading zeroes. YOLO `nc` is an exact positive
  integer class count and must equal the catalog length; `nc=10000` is valid while the largest
  class id remains 9999. Never parse `nc` through the class-id upper bound. Its YAML node must be
  an implicit plain scalar whose exact source slice is the canonical decimal; quotes, explicit
  tags, anchors, signs, padding, separators, and other coercible forms are invalid.
- `POST /projects/{project_id}/imports/{import_id}/complete` has one
  `avia.import-complete-queued/v1` response. A retry after response loss must decode the exact same
  persisted queued receipt, including non-empty `dispatch_mode` and `worker_task_id`; the client
  never weakens this contract or treats a partial replay response as success. Pending, uploaded,
  queued, running, and failed responses cannot expose a dataset-version identity; the client
  accepts only absent or paired-null identity fields there. A succeeded poll requires the exact
  ten fields emitted by `dataset_version_ref()`, materialized `minio_lakefs` storage, positive
  counts, canonical SHA-256 digest, complete lakeFS values, matching dataset id/tag, and a manifest
  path closed over its path prefix. Reduced, extra, empty, or internally inconsistent refs fail.
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
3.17's local backend, `woodpeckerci/plugin-git:2.9.2` is a clone plugin identifier, not an OCI
image pin, while every ordinary step `image` is a host executable and must be absolute
`/usr/bin/bash`. The first `host-toolchain` step hashes and executes the sole root-installed
`/usr/local/bin/avia-verify-woodpecker-local-toolchain` contract at the exact SHA-256
declared by the canonical workflow and executable contract test; it verifies the root
policy broker's exact pipeline/repository/event/commit/approval identity plus the complete host
toolchain. Do not duplicate a partial Bash/plugin-git verifier in this repository. Matrix
combinations are separate
workflows; inside each workflow, `host-toolchain -> quality -> package` is an explicit DAG so no
two steps mutate the same workspace `.venv`. Keep the clone entry free of explicit environment
variables: only that clone boundary may receive Woodpecker-injected `CI_NETRC_*`; the sole host
verifier dynamically rejects any non-empty `CI_NETRC_*` before repository code and prints only
the variable name. The server supplies a
GitHub-URL-scoped `GIT_CONFIG_*` proxy to `127.0.0.1:7897`, which does not proxy ordinary HTTP
clients. Native clone disables LFS and partial clone so checkout/reset cannot open a second
promisor-network fetch. Never restore `skip_clone` or a custom checkout script. Tracked sources must not be Git
LFS pointer files. Every step uses the same bootstrap-owned canonical PATH beginning with the
root-owned Node toolchain and system command directories. All repositories share
`UV_CACHE_DIR=/mnt/data/avia/cache/uv`, and this
same-device workspace must use `UV_LINK_MODE=hardlink`, never copy mode. The runner pre-provisions
all supported interpreters and the workflow sets `UV_PYTHON_DOWNLOADS=never`; CI must never fetch
a Python runtime implicitly. Compile-time warnings and pytest warnings owned by `avia_cli` or this
repository's tests fail the gate; third-party warnings remain visible and are never suppressed or
promoted by a global `-W error`. GitHub Actions is release-only for tags/manual trusted publishing
and runs the same frozen gates. Do not add hosted PR/main CI, server/GPU dependencies, or
deployment behavior to this public client repository.

The local backend is not a workload sandbox: the shared agent retains Docker authority, so an
approved malicious workflow could bypass repository-side commands. Human approval of the exact
commit is the security boundary; the root broker and verifier are fail-closed drift evidence, not
a replacement sandbox. Release acceptance must separately prove that the open PR head and the
Codex review refer to the same commit as the accepted Woodpecker pipeline.
