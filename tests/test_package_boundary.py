from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI_ROOT = ROOT / "packages" / "avia-cli"
CLI_SRC = CLI_ROOT / "src" / "avia_cli"

BANNED_IMPORT_PREFIXES = (
    "avia_sdk",
    "backend",
    "celery",
    "cleanvision",
    "eurekai_ai_platform",
    "fastapi",
    "fiftyone",
    "qdrant_client",
    "redis",
    "sqlalchemy",
    "torch",
    "ultralytics",
)

BANNED_METADATA_TERMS = (
    "avia-sdk",
    "torch",
    "ultralytics",
    "qdrant",
    "fiftyone",
    "fastapi",
    "sqlalchemy",
    "celery",
)


def test_repository_builds_one_distribution_with_avia_console_script() -> None:
    root_project = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    cli_project = (CLI_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    workflow = (ROOT / ".github" / "workflows" / "publish.yml").read_text(
        encoding="utf-8"
    )

    assert '"packages/avia-cli"' in root_project
    assert "packages/avia-sdk" not in root_project
    assert 'name = "avia-cli"' in cli_project
    assert 'avia = "avia_cli.main:main"' in cli_project
    assert 'packages = ["src/avia_cli"]' in cli_project
    assert "avia-sdk" not in cli_project
    assert "uv build --package avia-cli" in workflow
    assert "uv build --package avia-sdk" not in workflow
    assert not (ROOT / "packages" / "avia-sdk").exists()


def test_distribution_metadata_does_not_pull_server_or_algorithm_dependencies() -> None:
    pyproject = (CLI_ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()

    for term in BANNED_METADATA_TERMS:
        assert term not in pyproject


def test_package_does_not_import_server_algorithm_or_sdk_distribution_modules() -> None:
    imported: list[str] = []
    for path in sorted(CLI_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)

    assert imported
    for name in imported:
        assert not name.startswith(BANNED_IMPORT_PREFIXES), name
