# Avia CLI

Command line interface for Avia dataset upload and browser authentication.

Install:

```bash
pipx install avia-cli
avia --help
```

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
  --format yolo
```

The published distribution is intentionally a single package. Internal modules
under `avia_cli.core` keep auth and upload code reusable without exposing a
separate SDK package before that API is stable.
