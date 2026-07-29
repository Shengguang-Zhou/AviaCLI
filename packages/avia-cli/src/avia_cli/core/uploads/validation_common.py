from __future__ import annotations

import math
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path
from typing import Any

from PIL import Image

from avia_cli.core.uploads.manifest import is_client_state_path

_DOCUMENT_NAMES = {
    "LICENSE",
    "LICENSE.md",
    "LICENSE.txt",
    "README.md",
    "README.txt",
    "SOURCES.md",
    "license.txt",
    "provenance.json",
    "provenance.yaml",
    "provenance.yml",
    "source_records.json",
}


def error(code: str, message: str, **details: object) -> dict[str, Any]:
    return {"code": code, **details, "message": message}


def finite_numbers(values: list[object]) -> list[float] | None:
    try:
        numbers = [float(value) for value in values]
    except (TypeError, ValueError):
        return None
    return numbers if all(math.isfinite(value) for value in numbers) else None


def json_finite_numbers(value: object) -> list[float] | None:
    if not isinstance(value, list) or any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in value
    ):
        return None
    try:
        numbers = [float(item) for item in value]
    except OverflowError:
        return None
    return numbers if all(math.isfinite(item) for item in numbers) else None


def image_size(path: Path) -> tuple[int, int]:
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            width, height = int(image.width), int(image.height)
    except Exception as exc:
        raise ValueError(f"cannot fully decode image {path}: {exc}") from exc
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    return width, height


def dataset_role_directories(*, source_root: Path, role_root: Path) -> list[Path]:
    if not role_root.is_dir():
        return []
    return sorted(
        path
        for path in role_root.iterdir()
        if path.is_dir() and not is_client_state_path(path.relative_to(source_root))
    )


def polygon_area(points: list[tuple[float, float]]) -> float:
    return (
        abs(
            sum(
                x1 * y2 - x2 * y1
                for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1], strict=True)
            )
        )
        / 2.0
    )


def normalized_points(values: list[float]) -> list[tuple[float, float]] | None:
    if len(values) % 2 != 0 or any(value < 0.0 or value > 1.0 for value in values):
        return None
    return list(zip(values[::2], values[1::2], strict=True))


def is_document_path(relative_path: str) -> bool:
    return "/" not in relative_path and relative_path in _DOCUMENT_NAMES


def is_cache_path(relative_path: str) -> bool:
    parts = Path(relative_path).parts
    return any(
        part.lower()
        in {
            ".avia",
            ".cache",
            ".git",
            ".ipynb_checkpoints",
            ".pytest_cache",
            "__macosx",
            "__pycache__",
            "node_modules",
        }
        or part.endswith((".cache", ".pyc", ".tmp", "~"))
        for part in parts
    ) or parts[-1].lower() in {".ds_store", "thumbs.db"}


Point = tuple[float, float]
Edge = tuple[Point, Point]
_GEOMETRY_EPSILON = 1e-12


def is_weakly_simple_polygon(points: list[Point]) -> bool:
    """Validate simple rings joined only by canonical zero-width bridge trees."""

    canonical: list[Point] = []
    for point in points:
        if not canonical or canonical[-1] != point:
            canonical.append(point)
    if len(canonical) > 1 and canonical[-1] == canonical[0]:
        canonical.pop()
    if len(canonical) < 3:
        return False
    edges = list(zip(canonical, canonical[1:] + canonical[:1], strict=True))
    for index, (a1, a2) in enumerate(edges):
        for other_index, (b1, b2) in enumerate(edges):
            if other_index <= index:
                continue
            if other_index in {index + 1, (index - 1) % len(edges)}:
                continue
            if index == 0 and other_index == len(edges) - 1:
                continue
            if _segments_cross_properly(a1, a2, b1, b2):
                return False
    atomic_edges = _split_edges_at_vertices(edges, vertices=set(canonical))
    edge_counts = Counter(atomic_edges)
    if any(count != 1 for count in edge_counts.values()):
        return False
    bridge_edges = {edge for edge in edge_counts if (edge[1], edge[0]) in edge_counts}
    ring_edges = [edge for edge in atomic_edges if edge not in bridge_edges]
    rings = _decompose_directed_rings(ring_edges)
    if rings is None:
        return False
    return _bridges_form_tree(bridge_edges=bridge_edges, rings=rings)


def _orientation(first: Point, second: Point, third: Point) -> float:
    return (second[0] - first[0]) * (third[1] - first[1]) - (second[1] - first[1]) * (
        third[0] - first[0]
    )


def _segments_cross_properly(
    a1: Point,
    a2: Point,
    b1: Point,
    b2: Point,
) -> bool:
    def side(value: float) -> int:
        if value > _GEOMETRY_EPSILON:
            return 1
        if value < -_GEOMETRY_EPSILON:
            return -1
        return 0

    return (
        side(_orientation(a1, a2, b1)) * side(_orientation(a1, a2, b2)) == -1
        and side(_orientation(b1, b2, a1)) * side(_orientation(b1, b2, a2)) == -1
    )


def _split_edges_at_vertices(edges: list[Edge], *, vertices: set[Point]) -> list[Edge]:
    result: list[Edge] = []
    for start, end in edges:
        points = sorted(
            (point for point in vertices if _point_on_segment(point, start, end)),
            key=lambda point: _edge_parameter(point, start, end),
        )
        result.extend(pairwise(points))
    return result


def _point_on_segment(point: Point, start: Point, end: Point) -> bool:
    return (
        abs(_orientation(start, end, point)) <= _GEOMETRY_EPSILON
        and min(start[0], end[0]) - _GEOMETRY_EPSILON
        <= point[0]
        <= max(start[0], end[0]) + _GEOMETRY_EPSILON
        and min(start[1], end[1]) - _GEOMETRY_EPSILON
        <= point[1]
        <= max(start[1], end[1]) + _GEOMETRY_EPSILON
    )


def _edge_parameter(point: Point, start: Point, end: Point) -> float:
    if abs(end[0] - start[0]) >= abs(end[1] - start[1]):
        return (point[0] - start[0]) / (end[0] - start[0])
    return (point[1] - start[1]) / (end[1] - start[1])


def _decompose_directed_rings(edges: list[Edge]) -> list[list[Point]] | None:
    if not edges:
        return None
    outgoing: dict[Point, list[Point]] = defaultdict(list)
    incoming: dict[Point, list[Point]] = defaultdict(list)
    for start, end in edges:
        outgoing[start].append(end)
        incoming[end].append(start)
    vertices = set(outgoing) | set(incoming)
    if any(len(outgoing[point]) != 1 or len(incoming[point]) != 1 for point in vertices):
        return None

    unvisited = set(edges)
    rings: list[list[Point]] = []
    while unvisited:
        start = next(iter(unvisited))[0]
        current = start
        ring: list[Point] = []
        while True:
            if current in ring:
                return None
            ring.append(current)
            edge = (current, outgoing[current][0])
            if edge not in unvisited:
                return None
            unvisited.remove(edge)
            current = edge[1]
            if current == start:
                break
        if len(ring) < 3 or polygon_area(ring) <= _GEOMETRY_EPSILON:
            return None
        rings.append(ring)
    return rings


def _bridges_form_tree(*, bridge_edges: set[Edge], rings: list[list[Point]]) -> bool:
    if not bridge_edges:
        return len(rings) == 1
    ring_by_point = {point: ring_index for ring_index, ring in enumerate(rings) for point in ring}
    bridge_points = {point for edge in bridge_edges for point in edge if point not in ring_by_point}
    point_nodes = {point: len(rings) + index for index, point in enumerate(sorted(bridge_points))}

    def node(point: Point) -> int:
        if point in ring_by_point:
            return ring_by_point[point]
        return point_nodes[point]

    undirected_edges = {frozenset(edge) for edge in bridge_edges}
    contracted_edges: set[frozenset[int]] = set()
    for edge in undirected_edges:
        first, second = tuple(edge)
        contracted = frozenset({node(first), node(second)})
        if len(contracted) != 2 or contracted in contracted_edges:
            return False
        contracted_edges.add(contracted)
    nodes = set(range(len(rings))) | set(point_nodes.values())
    if len(contracted_edges) != len(nodes) - 1:
        return False
    adjacency: dict[int, set[int]] = defaultdict(set)
    for edge in contracted_edges:
        first, second = tuple(edge)
        adjacency[first].add(second)
        adjacency[second].add(first)
    reached: set[int] = set()
    pending = [next(iter(nodes))]
    while pending:
        current = pending.pop()
        if current in reached:
            continue
        reached.add(current)
        pending.extend(adjacency[current] - reached)
    return reached == nodes
