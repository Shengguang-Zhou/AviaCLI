from __future__ import annotations

from pathlib import Path
from typing import Any

_YOLO_META_FILES = ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml", "classes.txt")


def read_yolo_class_names(source_root: str | Path) -> list[str]:
    root = Path(source_root).expanduser().resolve()
    for name in _YOLO_META_FILES:
        candidate = root / name
        if not candidate.exists() or not candidate.is_file():
            continue
        if candidate.suffix.lower() == ".txt":
            return _read_classes_txt(candidate)
        return _read_yolo_yaml(candidate)
    return []


def _read_classes_txt(path: Path) -> list[str]:
    labels = [
        line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    if not labels:
        raise SystemExit(f"{path.name} is empty")
    return labels


def _read_yolo_yaml(path: Path) -> list[str]:
    try:
        import yaml  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency is installed in runtime/test envs
        raise RuntimeError("pyyaml is required to parse YOLO dataset metadata") from exc

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"invalid YOLO metadata YAML: {path}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"invalid YOLO metadata format: {path}")

    names_obj = payload.get("names")
    if names_obj is None:
        return []
    labels = _normalize_names(names_obj, path=path)
    nc_raw = payload.get("nc")
    if nc_raw is not None:
        try:
            nc = int(nc_raw)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"invalid YOLO nc in {path}") from exc
        if nc != len(labels):
            raise SystemExit(f"YOLO metadata nc mismatch in {path}: nc={nc}, names={len(labels)}")
    return labels


def _normalize_names(value: Any, *, path: Path) -> list[str]:
    if isinstance(value, list):
        labels = [str(item).strip() for item in value if str(item).strip()]
    elif isinstance(value, dict):
        pairs: list[tuple[int, str]] = []
        for key, raw_name in value.items():
            try:
                index = int(key)
            except (TypeError, ValueError) as exc:
                raise SystemExit(f"YOLO names keys must be integer class ids in {path}") from exc
            name = str(raw_name).strip()
            if name:
                pairs.append((index, name))
        labels = [name for _index, name in sorted(pairs)]
    else:
        raise SystemExit(f"YOLO names must be a list or dict in {path}")
    if not labels:
        raise SystemExit(f"YOLO names are empty in {path}")
    return labels
