# Avia CLI

Command line interface for Avia dataset upload and browser authentication.

Install:

```bash
pipx install avia-cli
avia --help
```

AviaCLI supports Python 3.10, 3.11, and 3.12.

One-off run:

```bash
uvx --from avia-cli avia --help
```

Login:

```bash
avia auth login --api https://avia.eurekailab.com/api/v1
```

Upload:

```bash
avia dataset upload \
  --project proj_123abc456def \
  --source /data/leonardo-airborne-object-recognition-yolo \
  --format yolo \
  --task-key detect
```

Inspect and verify a local dataset before upload:

```bash
avia dataset inspect \
  --source /data/coco8 \
  --format yolo \
  --task-key detect \
  --json

avia dataset verify \
  --source /data/coco8 \
  --format yolo \
  --task-key detect \
  --json
```

`--format` and `--task-key` are required for dataset scan, source import,
folder upload, inspect, and verify. There is no task inference
and no detection default. The accepted matrix is exact:

| Format | Tasks |
| --- | --- |
| `yolo` | `detect`, `classify`, `segment`, `pose`, `obb` |
| `coco` | `detect`, `segment`, `pose` |
| `imagenet` | `classify` |
| `anomalib` | `ad` |

Examples for the six task families:

```bash
avia dataset verify --source /data/coco8 --format yolo --task-key detect
avia dataset verify --source /data/multilabel --format yolo --task-key classify
avia dataset verify --source /data/coco8-seg --format yolo --task-key segment
avia dataset verify --source /data/coco8-pose --format yolo --task-key pose
avia dataset verify --source /data/dota8-strict --format yolo --task-key obb
avia dataset verify --source /data/avia/validation/avr-six-task-v2/ad --format anomalib --task-key ad
```

Registering an existing object-store prefix uses one canonical URI form: a bare NFC POSIX object
path with no leading slash and with a trailing slash. Scheme aliases, absolute paths, traversal,
backslashes, whitespace, and missing trailing slashes are rejected before authentication or HTTP:

```bash
avia import create \
  --project proj_123abc456def \
  --source datasets/coco8/ \
  --format yolo \
  --task-key detect
```

Validation fully decodes every image and rejects all symbolic links in the source tree, including
broken links and linked directories. The shared scanner also rejects FIFOs, sockets, devices, and
every other non-regular member instead of silently omitting it. Inspection, verification, and
upload always process the complete dataset through folder sessions.

After validation, each upload is bound to the same regular-file identity used to compute its
SHA-256 and dimensions. Hashing and PUT retries use an `O_NOFOLLOW`
descriptor and fail if the path, inode, size, or timestamps change. Presigned URLs returned by
the Avia API are used exactly as issued; the CLI has no origin rewrite, Host override, or hidden
alternate transport. Folder PUTs use one case-insensitive required-header validator: `Host` and
`Transfer-Encoding` are forbidden, `Content-Length` must match the verified file exactly, and
only typed transport failures or HTTP 408/429/5xx are retried. CPU parameters are derived locally;
transport concurrency probes the first API-issued storage URL, never the control-plane host.

Folder upload creates a canonical UUIDv4 idempotency key and atomically writes the sole current
pending-session state before the first HTTP request. If the response is lost, rerun the identical
command with `--resume`; AviaCLI replays the exact key and request. Historical state formats are
rejected so stale behavior cannot pass silently. Resume also requires the exact original file
set and regular-file identities, then rehashes every file already marked uploaded and compares
its SHA-256 before any network request.

YOLO validation is task-specific: exact row widths, finite normalized values,
declared class bounds, runtime-valid segment masks, OBB geometry, and pose `kpt_shape` are enforced.
Segment topology follows the AviaTraining/Ultralytics consumer: canonical thin bridges from
`merge_multi_segment` are accepted, while rasterizable crossings or non-canonical bridge topology
produce structured `yolo_segment_topology` warnings with exact file and line instead of rejecting
official training data.
Classification label files contain one unique class id per row. Anomalib
validation requires exactly `train/good`, `val/{good,bad}`, `test/{good,bad}`,
and one `ground_truth/{val,test}/bad/<same-stem>.png` mask per bad evaluation
sample. Role images are direct children and use lowercase `.jpg`, `.jpeg`, `.png`,
or `.webp`; masks use lowercase `.png`. The decoded encoding must match that suffix
exactly; renamed BMP/TIFF data is rejected, and masks must decode as PNG. YOLO,
COCO, and ImageNet preserve supported uppercase image suffixes and validate the
decoded encoding against their case-normalized suffix. ImageNet builds one
`(split, class)` index before class validation. Original MVTec
defect-name directories, `validation`, `_mask` suffix adaptation, missing roles,
and nested role directories are rejected. Root documentation is limited to
README, LICENSE, and `source_records.json`, matching the importer. Validation
reports the sole binary taxonomy `["good", "bad"]`. The same role inventory
drives inspection, validation, manifest generation, and upload JSON: source
samples are `image_count`, YOLO/COCO annotations are `label_count`, and
Anomalib `ground_truth/**` members are exclusively `mask_count`.
Validation errors exit non-zero; missing labels or unknown classes are never warnings.
For multilabel classification, an existing empty label file is an explicit
negative sample; an absent label file remains an error.

COCO polygon and RLE segmentation are both rasterized with `pycocotools`, then must produce one
connected mask without holes. The unique full contour is filled back into a mask and must equal
the source mask pixel-for-pixel, proving it can be represented losslessly by one YOLO polygon.
Exact cross-split taxonomy, image coverage, duplicate split membership, pose metadata, and
bounding-box corners are also validated. Anomalib masks must be non-empty binary single-channel
images with the same dimensions as their defect image.

API bases are canonical absolute `http(s)` URLs without credentials, queries, fragments,
whitespace, default ports, or trailing slashes. Session, signed-URL, batch-complete, finalization,
and poll responses use exact field and status decoders; historical aliases are rejected.
Signed media types are exact lowercase ASCII `token/token` values without parameters,
whitespace, controls, empty tokens, or extra slashes. Pre-publication responses cannot expose a
dataset-version identity, while a succeeded poll requires matching `dataset_version_id` and
`version_ref.dataset_version_id`. Auth
refresh occurs only for `401` or explicit `token_expired`, never `403`, and an explicit token is
never associated with environment password or refresh-token credentials from another identity.

Generate a cleanup plan from Avia server state and local resume files:

```bash
avia dataset cleanup-plan \
  --api https://avia.eurekailab.com/api/v1 \
  --project proj_123abc456def \
  --source /data/coco8 \
  --json
```

The CLI never connects to MinIO or lakeFS directly. Uploads, reuse checks,
server-side cleanup, and dataset version ownership stay behind the Avia API.

The published distribution is intentionally a single package. Internal modules
under `avia_cli.core` keep auth and upload code reusable without exposing a
separate SDK package before that API is stable.

Internal PR/manual CI uses Woodpecker 3.14's local backend, where
`woodpeckerci/plugin-git:2.9.2` is the host clone plugin identifier rather than an OCI image pin.
The complete shallow fetch disables LFS and partial clone. After the commit is fetched, checkout
cannot open a second promisor-remote TLS request. Clone-only NETRC credentials are rejected by
every ordinary `/usr/bin/bash` step. The workflow shares
`/mnt/data/avia/cache/uv` with the AVR repositories and uses same-device hardlinks; custom checkout
scripts, per-repository uv caches, and copy mode are not part of the CI contract. The root project
pins uv 0.8.3; separate Python 3.10, 3.11, and 3.12 matrix workflows run owned-warning
quality gates before the Python 3.12 workflow builds the package once.
