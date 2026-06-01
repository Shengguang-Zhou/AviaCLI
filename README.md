# Avia Python Client

This repository contains the public Python client packages for Avia.

## Packages

- `avia-sdk`: reusable Python SDK for Avia public HTTP APIs, auth helpers,
  dataset manifest scanning, signed uploads, and import-session operations.
- `avia-cli`: command line interface that depends on `avia-sdk` and owns only
  parsing, terminal output, and local credential storage.

## Install

For users:

```bash
pipx install avia-cli
avia --help
```

For one-off use:

```bash
uvx --from avia-cli avia --help
```

For Python code:

```bash
pip install avia-sdk
```

## Development

```bash
uv sync --all-packages --group dev
uv run pytest -q
uv run ruff check packages tests
uv build --package avia-sdk
uv build --package avia-cli
```

## Boundary

This repo may contain public client code only. Do not add backend application
imports, Runtime worker code, database models, Celery tasks, GPU/model
libraries, vector store clients, or curation/quality algorithm implementations.

## Release

Releases use PyPI Trusted Publishing through GitHub Actions.

1. Configure Trusted Publishers for `avia-sdk` and `avia-cli` on TestPyPI and
   PyPI with repository `Shengguang-Zhou/AviaCLI` and workflow
   `.github/workflows/publish.yml`.
2. Run the manual workflow for TestPyPI.
3. Verify TestPyPI install.
4. Push a version tag such as `v0.1.0` to publish both packages to PyPI.
