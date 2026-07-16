from __future__ import annotations

from pathlib import Path
from typing import Any

_YOLO_META_FILES = ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml", "classes.txt")


def read_yolo_class_names(source_root: str | Path) -> list[str]:
    return [str(name) for name in list(read_yolo_metadata(source_root).get("names") or [])]


def read_yolo_metadata(source_root: str | Path) -> dict[str, object]:
    root = Path(source_root).expanduser().resolve()
    candidates = [root / name for name in _YOLO_META_FILES if (root / name).is_file()]
    if not candidates:
        return {}
    if len(candidates) != 1:
        names = [path.name for path in candidates]
        raise SystemExit(f"YOLO metadata must have exactly one source of truth: {names}")
    candidate = candidates[0]
    if candidate.suffix.lower() == ".txt":
        return {"names": _read_classes_txt(candidate)}
    return _read_yolo_yaml(candidate)


def _read_classes_txt(path: Path) -> list[str]:
    labels = path.read_text(encoding="utf-8").splitlines()
    if not labels or any(not label or label != label.strip() for label in labels):
        raise SystemExit(f"YOLO names must be canonical non-empty strings in {path}")
    if len(set(labels)) != len(labels):
        raise SystemExit(f"YOLO names must be unique in {path}")
    return labels


def _read_yolo_yaml(path: Path) -> dict[str, object]:
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
    labels = _normalize_names(names_obj, path=path) if names_obj is not None else []
    nc_raw = payload.get("nc")
    if nc_raw is not None:
        try:
            nc = int(nc_raw)
        except (TypeError, ValueError) as exc:
            raise SystemExit(f"invalid YOLO nc in {path}") from exc
        if nc != len(labels):
            raise SystemExit(f"YOLO metadata nc mismatch in {path}: nc={nc}, names={len(labels)}")
    metadata: dict[str, object] = {"names": labels}
    if "kpt_shape" in payload:
        metadata["kpt_shape"] = payload["kpt_shape"]
    return metadata


def _normalize_names(value: Any, *, path: Path) -> list[str]:
    if isinstance(value, list):
        if any(not isinstance(item, str) or not item or item != item.strip() for item in value):
            raise SystemExit(f"YOLO names must contain canonical non-empty strings in {path}")
        labels = list(value)
    elif isinstance(value, dict):
        pairs: list[tuple[int, str]] = []
        for key, raw_name in value.items():
            try:
                index = int(key)
            except (TypeError, ValueError) as exc:
                raise SystemExit(f"YOLO names keys must be integer class ids in {path}") from exc
            if not isinstance(raw_name, str) or not raw_name or raw_name != raw_name.strip():
                raise SystemExit(f"YOLO names must contain canonical non-empty strings in {path}")
            name = raw_name
            pairs.append((index, name))
        ordered = sorted(pairs)
        if [index for index, _name in ordered] != list(range(len(ordered))):
            raise SystemExit(f"YOLO names class ids must be contiguous from zero in {path}")
        labels = [name for _index, name in ordered]
    else:
        raise SystemExit(f"YOLO names must be a list or dict in {path}")
    if not labels:
        raise SystemExit(f"YOLO names are empty in {path}")
    if len(set(labels)) != len(labels):
        raise SystemExit(f"YOLO names must be unique in {path}")
    return labels
