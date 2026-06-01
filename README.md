# Avia Python Client

This repository contains the public Python command line client for Avia.

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

For one-off use:

```bash
uvx --from avia-cli avia --help
```

## Development

```bash
uv sync --all-packages --group dev
uv run pytest -q
uv run ruff check packages tests
uv build --package avia-cli
```

## Boundary

This repo may contain public client code only. Do not add backend application
imports, Runtime worker code, database models, Celery tasks, GPU/model
libraries, vector store clients, or curation/quality algorithm implementations.

## Release

Releases use PyPI Trusted Publishing through GitHub Actions.

1. Configure the Trusted Publisher for `avia-cli` on TestPyPI and PyPI with
   repository `Shengguang-Zhou/AviaCLI`, workflow `publish.yml`, and
   environment `pypi`.
2. Run the manual workflow for TestPyPI.
3. Verify TestPyPI install.
4. Push a version tag such as `v0.1.0` to publish the package to PyPI.
