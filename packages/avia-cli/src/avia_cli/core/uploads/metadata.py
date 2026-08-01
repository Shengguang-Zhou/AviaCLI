from __future__ import annotations

from pathlib import Path
from avia_cli.core.uploads.class_catalog import (
    require_canonical_class_catalog,
    require_canonical_class_index,
    require_class_count,
    require_indexed_class_catalog,
)

_YOLO_META_FILES = ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml", "classes.txt")


def read_yolo_class_names(source_root: str | Path) -> list[str]:
    return list(read_yolo_metadata(source_root).get("names") or [])


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
    content = path.read_text(encoding="utf-8")
    if content.endswith("\n"):
        content = content[:-1]
    labels = content.split("\n")
    try:
        return require_canonical_class_catalog(labels, label=f"YOLO names in {path}")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _read_yolo_yaml(path: Path) -> dict[str, object]:
    try:
        import yaml  # type: ignore
        from yaml.nodes import MappingNode, ScalarNode, SequenceNode  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency is installed in runtime/test envs
        raise RuntimeError("pyyaml is required to parse YOLO dataset metadata") from exc

    try:
        content = path.read_text(encoding="utf-8")
        root = yaml.compose(content, Loader=yaml.SafeLoader)
        payload = yaml.safe_load(content)
    except Exception as exc:
        raise SystemExit(f"invalid YOLO metadata YAML: {path}") from exc
    if not isinstance(root, MappingNode) or not isinstance(payload, dict):
        raise SystemExit(f"invalid YOLO metadata format: {path}")

    identity_nodes: dict[str, object] = {}
    for key_node, value_node in root.value:
        if not isinstance(key_node, ScalarNode) or key_node.tag != "tag:yaml.org,2002:str":
            continue
        key = key_node.value
        if key not in {"names", "nc"}:
            continue
        if key in identity_nodes:
            raise SystemExit(f"duplicate YOLO metadata {key} in {path}")
        identity_nodes[key] = value_node

    names_node = identity_nodes.get("names")
    labels = (
        _class_names_from_yaml_node(
            names_node,
            path=path,
            mapping_node_type=MappingNode,
            scalar_node_type=ScalarNode,
            sequence_node_type=SequenceNode,
        )
        if names_node is not None
        else []
    )
    nc_node = identity_nodes.get("nc")
    if nc_node is not None:
        if not isinstance(nc_node, ScalarNode) or nc_node.tag != "tag:yaml.org,2002:int":
            raise SystemExit(f"invalid YOLO nc in {path}")
        try:
            nc = require_class_count(
                require_canonical_class_index(
                    nc_node.value,
                    label=f"YOLO nc in {path}",
                ),
                label=f"YOLO nc in {path}",
                allow_zero=False,
            )
        except ValueError as exc:
            raise SystemExit(f"invalid YOLO nc in {path}") from exc
        if nc != len(labels):
            raise SystemExit(f"YOLO metadata nc mismatch in {path}: nc={nc}, names={len(labels)}")
    metadata: dict[str, object] = {"names": labels}
    if "kpt_shape" in payload:
        metadata["kpt_shape"] = payload["kpt_shape"]
    return metadata


def _class_names_from_yaml_node(
    value: object,
    *,
    path: Path,
    mapping_node_type: type,
    scalar_node_type: type,
    sequence_node_type: type,
) -> list[str]:
    if isinstance(value, sequence_node_type):
        labels = [
            _yaml_class_name(
                node,
                path=path,
                label=f"YOLO names[{index}]",
                scalar_node_type=scalar_node_type,
            )
            for index, node in enumerate(value.value)
        ]
    elif isinstance(value, mapping_node_type):
        labels_by_index: dict[object, str] = {}
        seen_indices: set[int] = set()
        for index_node, name_node in value.value:
            if not isinstance(index_node, scalar_node_type) or index_node.tag not in {
                "tag:yaml.org,2002:int",
                "tag:yaml.org,2002:str",
            }:
                raise SystemExit(f"YOLO names keys must be canonical class ids in {path}")
            try:
                index = require_canonical_class_index(
                    index_node.value,
                    label=f"YOLO names index in {path}",
                )
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            if index in seen_indices:
                raise SystemExit(f"YOLO names class ids must be unique in {path}")
            seen_indices.add(index)
            raw_key: object = (
                index if index_node.tag == "tag:yaml.org,2002:int" else index_node.value
            )
            labels_by_index[raw_key] = _yaml_class_name(
                name_node,
                path=path,
                label=f"YOLO names[{index}]",
                scalar_node_type=scalar_node_type,
            )
        labels = labels_by_index
    else:
        raise SystemExit(f"YOLO names must be a list or dict in {path}")
    try:
        return require_indexed_class_catalog(labels, label=f"YOLO names in {path}")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _yaml_class_name(
    value: object,
    *,
    path: Path,
    label: str,
    scalar_node_type: type,
) -> str:
    if (
        not isinstance(value, scalar_node_type)
        or value.tag != "tag:yaml.org,2002:str"
        or type(value.value) is not str
    ):
        raise SystemExit(f"{label} must be a string in {path}")
    return value.value
