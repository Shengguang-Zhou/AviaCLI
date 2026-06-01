from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SDK_SRC = ROOT / "packages" / "avia-sdk" / "src" / "avia_sdk"
CLI_SRC = ROOT / "packages" / "avia-cli" / "src" / "avia_cli"

BANNED_IMPORT_PREFIXES = (
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
    "torch",
    "ultralytics",
    "qdrant",
    "fiftyone",
    "fastapi",
    "sqlalchemy",
    "celery",
)


def test_distribution_metadata_exposes_avia_console_script() -> None:
    sdk_pyproject = (ROOT / "packages" / "avia-sdk" / "pyproject.toml").read_text(
        encoding="utf-8"
    )
    cli_pyproject = (ROOT / "packages" / "avia-cli" / "pyproject.toml").read_text(
        encoding="utf-8"
    )

    assert 'name = "avia-sdk"' in sdk_pyproject
    assert 'name = "avia-cli"' in cli_pyproject
    assert '"avia-sdk==0.1.0"' in cli_pyproject
    assert 'avia = "avia_cli.main:main"' in cli_pyproject
    assert 'packages = ["src/avia_sdk"]' in sdk_pyproject
    assert 'packages = ["src/avia_cli"]' in cli_pyproject


def test_distribution_metadata_does_not_pull_server_or_algorithm_dependencies() -> None:
    pyproject = "\n".join(
        [
            (ROOT / "packages" / "avia-sdk" / "pyproject.toml").read_text(
                encoding="utf-8"
            ),
            (ROOT / "packages" / "avia-cli" / "pyproject.toml").read_text(
                encoding="utf-8"
            ),
        ]
    ).lower()

    for term in BANNED_METADATA_TERMS:
        assert term not in pyproject


def test_package_does_not_import_server_or_algorithm_modules() -> None:
    imported: list[str] = []
    for path in [*sorted(SDK_SRC.rglob("*.py")), *sorted(CLI_SRC.rglob("*.py"))]:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)

    assert imported
    for name in imported:
        assert not name.startswith(BANNED_IMPORT_PREFIXES), name
