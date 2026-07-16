from pathlib import Path


def test_workspace_and_package_publish_one_installable_python_range() -> None:
    root = Path(__file__).resolve().parents[1]
    workspace = (root / "pyproject.toml").read_text(encoding="utf-8")
    package = (root / "packages" / "avia-cli" / "pyproject.toml").read_text(encoding="utf-8")

    assert 'requires-python = ">=3.10,<3.13"' in workspace
    assert '[tool.uv]\nrequired-version = "==0.8.3"' in workspace
    assert 'requires-python = ">=3.10,<3.13"' in package
    assert '"numpy>=1.24,<2"' in package
    for version in ("3.10", "3.11", "3.12"):
        assert f'"Programming Language :: Python :: {version}"' in package
