from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

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


def _isolated_pytest_environment() -> dict[str, str]:
    excluded = {"PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTHONWARNINGS"}
    environment = {name: value for name, value in os.environ.items() if name not in excluded}
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return environment


def _handler_records_bound_exception(handler: ast.ExceptHandler) -> bool:
    if handler.name is None:
        return False

    for node in ast.walk(handler):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.NamedExpr)):
            value = node.value
            if isinstance(value, ast.Name) and value.id == handler.name:
                return True
        if isinstance(node, ast.Call):
            arguments = [*node.args, *(keyword.value for keyword in node.keywords)]
            if any(
                isinstance(descendant, ast.Name) and descendant.id == handler.name
                for argument in arguments
                for descendant in ast.walk(argument)
            ):
                return True
    return False


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


def test_internal_pr_ci_uses_local_backend_shared_cache_and_runs_full_release_gates() -> None:
    workflow_path = ROOT / ".woodpecker" / "ci.yml"
    workflow_text = workflow_path.read_text(encoding="utf-8")
    workflow = yaml.safe_load(workflow_text)
    clone = workflow["clone"]["git"]
    steps = workflow["steps"]

    assert workflow["when"] == [{"event": ["pull_request", "manual"]}]
    assert workflow["labels"] == {"backend": "local"}
    assert clone == {
        "image": "woodpeckerci/plugin-git:2.9.2",
        "settings": {"lfs": False, "partial": False},
    }
    assert list(steps) == ["host-toolchain", "quality", "package"]
    assert all(step["image"] == "/usr/bin/bash" for step in steps.values())
    assert steps["host-toolchain"].get("depends_on") is None
    assert steps["quality"]["depends_on"] == ["host-toolchain"]
    assert steps["package"]["depends_on"] == ["quality"]
    assert "assert_no_ci_netrc.sh" not in workflow_text
    assert workflow["matrix"] == {"PYTHON_VERSION": ["3.10", "3.11", "3.12"]}
    assert steps["package"]["when"] == [{"matrix": {"PYTHON_VERSION": "3.12"}}]

    assert "http_proxy:" not in workflow_text
    assert "https_proxy:" not in workflow_text
    assert "no_proxy:" not in workflow_text
    assert "skip_clone" not in workflow_text
    assert "checkout_cached_source.sh" not in workflow_text
    assert "UV_CACHE_DIR: /mnt/data/avia/cache/uv" in workflow_text
    assert workflow_text.count("UV_PYTHON_INSTALL_DIR: /mnt/data/avia/python") == 2
    assert "UV_LINK_MODE: hardlink" in workflow_text
    assert workflow_text.count("UV_PYTHON_DOWNLOADS: never") == 2
    assert "UV_LINK_MODE: copy" not in workflow_text
    assert "/mnt/data/avia/cache/uv-avia-cli" not in workflow_text
    assert "uv sync --python ${PYTHON_VERSION} --frozen --all-packages --group dev" in workflow_text
    assert "uv run --python ${PYTHON_VERSION} pytest -p no:cacheprovider -q" in workflow_text
    assert "pytest -W error" not in workflow_text
    assert (
        "uv run --python ${PYTHON_VERSION} python -W error -m compileall -q -f "
        "packages/avia-cli/src/avia_cli tests"
    ) in workflow_text
    assert "uv run --python ${PYTHON_VERSION} ruff check packages tests" in workflow_text
    assert "uv run --python ${PYTHON_VERSION} ruff format --check packages tests" in workflow_text
    assert "uv build --package avia-cli" in workflow_text


def test_local_backend_uses_the_single_root_policy_verifier() -> None:
    workflow_text = (ROOT / ".woodpecker" / "ci.yml").read_text(encoding="utf-8")
    workflow = yaml.safe_load(workflow_text)
    host = workflow["steps"]["host-toolchain"]
    verifier = host["commands"]

    assert len(verifier) == 1
    assert "17b2610d5658a41bd5d60d6c1f865506ea68b7064fe1d0f39baabdaae7196b58" in verifier[0]
    assert "sha256sum --check --strict -" in verifier[0]
    assert (
        "/usr/bin/bash --noprofile --norc /usr/local/bin/avia-verify-woodpecker-local-toolchain"
    ) in verifier[0]
    assert "command -v bash" not in workflow_text
    assert "command -v plugin-git" not in workflow_text
    canonical_path = (
        "/opt/avia/toolchains/node-v22.12.0/bin:/usr/local/sbin:/usr/local/bin:"
        "/usr/sbin:/usr/bin:/sbin:/bin:/home/dbcloud/.local/bin:"
        "/mnt/data/avia/wheelhouse/bin"
    )
    assert all(step["environment"]["PATH"] == canonical_path for step in workflow["steps"].values())


def test_pytest_warning_policy_fails_owned_modules_and_reports_dependencies(tmp_path: Path) -> None:
    owned_probe = tmp_path / "test_owned_warning.py"
    dependency_probe = tmp_path / "test_dependency_warning.py"
    owned_probe.write_text(
        "import warnings\n\n"
        "import pytest\n\n"
        "@pytest.mark.parametrize(\n"
        "    'owned_module',\n"
        "    [\n"
        "        'avia_cli.warning_probe',\n"
        "        'tests.warning_probe',\n"
        "        'conftest',\n"
        "        'test_warning_probe',\n"
        "    ],\n"
        ")\n"
        "def test_owned_warning(owned_module):\n"
        "    warnings.warn_explicit(\n"
        "        f'owned warning probe: {owned_module}',\n"
        "        UserWarning, __file__, 1, module=owned_module,\n"
        "    )\n",
        encoding="utf-8",
    )
    dependency_probe.write_text(
        "import warnings\n\n"
        "def test_dependency_warning():\n"
        "    warnings.warn_explicit(\n"
        "        'dependency warning probe', ImportWarning, __file__, 1,\n"
        "        module='dependency.warning_probe',\n"
        "    )\n",
        encoding="utf-8",
    )
    environment = _isolated_pytest_environment()

    owned = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-q",
            "-c",
            str(ROOT / "pyproject.toml"),
            str(owned_probe),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=environment,
    )
    dependency = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-q",
            "-c",
            str(ROOT / "pyproject.toml"),
            str(dependency_probe),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=environment,
    )

    owned_output = owned.stdout + owned.stderr
    dependency_output = dependency.stdout + dependency.stderr

    assert owned.returncode != 0, owned_output
    assert "4 failed" in owned_output
    for module in (
        "avia_cli.warning_probe",
        "tests.warning_probe",
        "conftest",
        "test_warning_probe",
    ):
        assert f"owned warning probe: {module}" in owned_output
    assert dependency.returncode == 0, dependency_output
    assert "dependency warning probe" in dependency_output


def test_internal_ci_runs_quality_on_every_supported_python_and_builds_once() -> None:
    workflow = (ROOT / ".woodpecker" / "ci.yml").read_text(encoding="utf-8")

    assert 'PYTHON_VERSION: ["3.10", "3.11", "3.12"]' in workflow
    assert "uv sync --python ${PYTHON_VERSION}" in workflow
    assert "uv run --python ${PYTHON_VERSION} pytest -p no:cacheprovider -q" in workflow
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
    assert "pytest -W error" not in workflow
    assert "python -W error -m compileall -q -f packages/avia-cli/src/avia_cli tests" in workflow


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


def test_broad_exception_handlers_record_or_raise_every_failure() -> None:
    for path in sorted(CLI_SRC.rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for handler in (node for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler)):
            broad = handler.type is None or (
                isinstance(handler.type, ast.Name)
                and handler.type.id in {"Exception", "BaseException"}
            )
            if not broad:
                continue
            assert not any(isinstance(node, ast.Pass) for node in ast.walk(handler)), (
                f"silent broad exception handler: {path}:{handler.lineno}"
            )
            if any(isinstance(node, ast.Raise) for node in ast.walk(handler)):
                continue
            assert _handler_records_bound_exception(handler), (
                f"broad exception is neither raised nor recorded: {path}:{handler.lineno}"
            )


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
    assert "woodpeckerci/plugin-git:2.9.2" in agents
    assert "clone plugin identifier, not an OCI" in agents
    assert "ordinary step `image` is a host executable" in agents
    assert "only that clone boundary may receive" in agents
    assert "UV_LINK_MODE=hardlink" in agents
    assert "thin-bridge" in agents


def test_production_package_uses_only_aviacli_identity() -> None:
    source = "\n".join(path.read_text(encoding="utf-8") for path in sorted(CLI_SRC.rglob("*.py")))

    assert "Avia SDK" not in source


def test_package_readme_uses_the_canonical_clone_plugin_version() -> None:
    readme = (ROOT / "packages" / "avia-cli" / "README.md").read_text(encoding="utf-8")

    assert "plugin-git:2.9.2" in readme
    assert "plugin-git:2.8.0" not in readme
    assert "host clone plugin identifier rather than an OCI image pin" in readme
    assert "disables LFS and partial clone" in readme
    assert "second promisor-remote TLS request" in readme
    assert re.search(
        r"rejects\s+every nonempty clone-only `CI_NETRC_\*` variable",
        readme,
    )


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
