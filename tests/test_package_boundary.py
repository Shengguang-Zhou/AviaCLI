from __future__ import annotations

import ast
import re
import subprocess
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
    workflow = (ROOT / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8")

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


def test_internal_pr_ci_uses_native_clone_shared_cache_and_runs_full_release_gates() -> None:
    workflow = (ROOT / ".woodpecker" / "ci.yml").read_text(encoding="utf-8")

    assert "event: [pull_request, manual]" in workflow
    assert "backend: local" in workflow
    assert "image: woodpeckerci/plugin-git:2.8.0" in workflow
    assert "http_proxy: http://127.0.0.1:7897" in workflow
    assert "https_proxy: http://127.0.0.1:7897" in workflow
    assert "no_proxy: 127.0.0.1,localhost,192.168.1.9" in workflow
    assert "lfs: false" in workflow
    assert "skip_clone" not in workflow
    assert "checkout_cached_source.sh" not in workflow
    assert "UV_CACHE_DIR: /mnt/data/avia/cache/uv" in workflow
    assert "UV_LINK_MODE: hardlink" in workflow
    assert "UV_LINK_MODE: copy" not in workflow
    assert "/mnt/data/avia/cache/uv-avia-cli" not in workflow
    assert "uv sync --python ${PYTHON_VERSION} --frozen --all-packages --group dev" in workflow
    assert "uv run --python ${PYTHON_VERSION} pytest -W error" in workflow
    assert "uv run --python ${PYTHON_VERSION} ruff check packages tests" in workflow
    assert "uv run --python ${PYTHON_VERSION} ruff format --check packages tests" in workflow
    assert "uv build --package avia-cli" in workflow


def test_internal_ci_runs_quality_on_every_supported_python_and_builds_once() -> None:
    workflow = (ROOT / ".woodpecker" / "ci.yml").read_text(encoding="utf-8")

    assert 'PYTHON_VERSION: ["3.10", "3.11", "3.12"]' in workflow
    assert "uv sync --python ${PYTHON_VERSION}" in workflow
    assert "uv run --python ${PYTHON_VERSION} pytest -W error" in workflow
    assert workflow.count("uv build --package avia-cli") == 1
    assert 'PYTHON_VERSION: "3.12"' in workflow
    assert (ROOT / "pyproject.toml").read_text(encoding="utf-8").count(
        'required-version = "==0.8.3"'
    ) == 1
    assert not (ROOT / "uv.toml").exists()


def test_release_ci_tests_every_supported_python_then_builds_and_publishes_once() -> None:
    workflow = (ROOT / ".github" / "workflows" / "publish.yml").read_text(encoding="utf-8")

    assert "python-version: [\"3.10\", \"3.11\", \"3.12\"]" in workflow
    assert "python-version: ${{ matrix.python-version }}" in workflow
    assert workflow.count('version: "0.8.3"') == 2
    assert workflow.count("uv build --package avia-cli") == 1
    assert "needs: quality" in workflow
    assert workflow.count("needs: build") == 2


def test_woodpecker_shell_locals_are_escaped_until_shell_execution() -> None:
    workflow = (ROOT / ".woodpecker" / "ci.yml").read_text(encoding="utf-8")

    shell_locals = set(re.findall(r"^\s+([A-Za-z_][A-Za-z0-9_]*)=", workflow, re.MULTILINE))
    unescaped_locals = [
        name
        for name in sorted(shell_locals)
        if re.search(rf"(?<!\$)\$\{{{re.escape(name)}\}}", workflow)
    ]
    assert unescaped_locals == []
    assert "build_dir" in shell_locals


def test_tracked_sources_are_not_git_lfs_pointer_files() -> None:
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    pointer_header = b"version " + b"https://git-lfs.github.com/spec/v1"
    pointers: list[str] = []
    for raw in tracked:
        if not raw:
            continue
        relative = raw.decode("utf-8")
        path = ROOT / relative
        if path.is_file():
            with path.open("rb") as handle:
                if handle.read(len(pointer_header)) == pointer_header:
                    pointers.append(relative)
    assert pointers == []


def test_agents_document_is_living_source_of_truth_for_upload_invariants() -> None:
    agents = (ROOT / "AGENTS.md").read_text(encoding="utf-8")

    assert "idempotency_key" in agents
    assert "pending-session" in agents
    assert "symbolic links" in agents
    assert "Never truncate" in agents
    assert ".woodpecker/ci.yml" in agents
    assert "woodpeckerci/plugin-git:2.8.0" in agents
    assert "UV_LINK_MODE=hardlink" in agents
    assert "thin-bridge" in agents


def test_production_package_uses_only_aviacli_identity() -> None:
    source = "\n".join(path.read_text(encoding="utf-8") for path in sorted(CLI_SRC.rglob("*.py")))

    assert "Avia SDK" not in source


def test_upload_code_has_no_origin_or_host_rewrite_bypass() -> None:
    source = "\n".join(path.read_text(encoding="utf-8") for path in sorted(CLI_SRC.rglob("*.py")))
    dataset_source = (CLI_SRC / "core" / "uploads" / "dataset.py").read_text(encoding="utf-8")
    api_source = (CLI_SRC / "core" / "uploads" / "api.py").read_text(encoding="utf-8")

    assert "upload_url_origin_override" not in source
    assert "upload-url-origin-override" not in source
    assert "upload_request_from_api" not in source
    assert "_should_bypass_proxy_for_upload" not in source
    assert "AVIA_UPLOAD_NO_PROXY_HOSTS" not in source
    assert 'putheader("Host"' not in source
    assert "_AviaHTTPError =" not in dataset_source
    assert "_UploadHTTPError =" not in dataset_source
    assert "__all__" not in dataset_source
    assert "_UPLOAD_CHUNK_SIZE" not in dataset_source
    assert "_DEFAULT_UPLOAD_READ_TIMEOUT" not in dataset_source
    assert "_DEFAULT_UPLOAD_RETRY_BASE_DELAY" not in dataset_source
    assert "_IMPORT_POLL_FAST_DELAYS_SEC" not in dataset_source
    assert "_UPLOAD_CHUNK_SIZE" not in api_source
