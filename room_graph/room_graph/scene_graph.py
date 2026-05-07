"""Stage 2 — geometric scene graph and dense relation matrix."""

from __future__ import annotations

import math
from typing import Iterable

import networkx as nx
import numpy as np
from shapely.geometry import Point

from .config import GraphConfig
from .geometry import (
    EntityGeom,
    RoomGeometry,
    WallSegment,
    angle_between_deg,
    back_edge_distance_to_wall,
    back_edge_overlap_fraction,
    footprint_clearance_to_wall,
    relative_direction,
    yaw_diff_modpi_deg,
)


ROOM_NODE_ID = "room_0"

DENSE_FEATURE_NAMES: tuple[str, ...] = (
    "distance_m",
    "relative_angle_deg",
    "footprint_overlap_m2",
    "facing_score",
    "path_blocking_score",
    "same_cluster_score",
    "line_of_sight",
)


# ---------------------------------------------------------------------------
# Node creation
# ---------------------------------------------------------------------------


def _node_type_for_kind(kind: str) -> str:
    if kind in ("furniture", "object"):
        return "object"
    if kind in ("wall", "door", "window"):
        return kind
    return "unknown"


def _add_entity_nodes(graph: nx.MultiDiGraph, geom: RoomGeometry) -> None:
    for entity in geom.entities.values():
        graph.add_node(
            entity.id,
            type=_node_type_for_kind(entity.kind),
            kind=entity.kind,
            label=entity.label,
            geometry=entity.to_dict(),
        )


def _add_room_node(graph: nx.MultiDiGraph, geom: RoomGeometry) -> None:
    centroid = geom.room_polygon.centroid
    graph.add_node(
        ROOM_NODE_ID,
        type="room",
        kind="room",
        label="room",
        geometry={
            "area_m2": geom.room_polygon.area,
            "centroid_xy": [centroid.x, centroid.y],
        },
        attrs={
            "entry_door_ids": list(geom.entry_door_ids),
            "passage_door_ids": list(geom.passage_door_ids),
            "n_walls": sum(1 for e in geom.entities.values() if e.kind == "wall"),
        },
    )


# ---------------------------------------------------------------------------
# Pairwise features
# ---------------------------------------------------------------------------


def _facing_score(source: EntityGeom, target: EntityGeom) -> float:
    fx, fy = source.forward_axis_world()
    dx = target.cx - source.cx
    dy = target.cy - source.cy
    n = math.hypot(dx, dy)
    if n == 0:
        return 0.0
    return max(0.0, (dx * fx + dy * fy) / n)


def _path_blocking_score(
    source: EntityGeom, target: EntityGeom, room_geom: RoomGeometry
) -> float:
    """Heuristic: max footprint of any non-soft furniture intersecting a corridor.

    The corridor is the line from source center to target center buffered by 0.3 m.
    """

    if source.id == target.id:
        return 0.0
    from shapely.geometry import LineString

    line = LineString([(source.cx, source.cy), (target.cx, target.cy)])
    if line.length <= 1e-6:
        return 0.0
    corridor = line.buffer(0.30)
    score = 0.0
    for other in room_geom.entities.values():
        if other.id in (source.id, target.id):
            continue
        if other.kind not in ("object", "furniture"):
            continue
        if other.is_soft():
            continue
        inter = corridor.intersection(other.footprint())
        if inter.is_empty:
            continue
        score = max(score, inter.area)
    return score


def _pair_features(
    source: EntityGeom,
    target: EntityGeom,
    room_geom: RoomGeometry,
) -> dict[str, float]:
    dx = target.cx - source.cx
    dy = target.cy - source.cy
    distance = math.hypot(dx, dy)
    rel_angle = yaw_diff_modpi_deg(source.yaw_rad, target.yaw_rad)
    overlap = source.footprint().intersection(target.footprint()).area
    facing = _facing_score(source, target)
    path_block = _path_blocking_score(source, target, room_geom)
    los = float(room_geom.line_of_sight(source.center, target.center))
    return {
        "distance_m": float(distance),
        "relative_angle_deg": float(rel_angle),
        "footprint_overlap_m2": float(overlap),
        "facing_score": float(facing),
        "path_blocking_score": float(path_block),
        "same_cluster_score": 0.0,  # filled in by functional stage
        "line_of_sight": los,
    }


def build_dense_relation_matrix(
    room_geom: RoomGeometry,
) -> tuple[list[str], np.ndarray]:
    """Compute the dense per-pair feature tensor for furniture entities.

    Returns `(id_order, matrix)` where `matrix.shape == (N, N, len(DENSE_FEATURE_NAMES))`.
    """

    furniture = sorted(room_geom.furniture(), key=lambda e: e.id)
    id_order = [e.id for e in furniture]
    n = len(furniture)
    k = len(DENSE_FEATURE_NAMES)
    mat = np.zeros((n, n, k), dtype=np.float64)
    for i, src in enumerate(furniture):
        for j, dst in enumerate(furniture):
            if i == j:
                continue
            feats = _pair_features(src, dst, room_geom)
            for fi, name in enumerate(DENSE_FEATURE_NAMES):
                mat[i, j, fi] = feats[name]
    return id_order, mat


# ---------------------------------------------------------------------------
# Edge emission
# ---------------------------------------------------------------------------


def _emit_inside_room_edges(graph: nx.MultiDiGraph, room_geom: RoomGeometry) -> None:
    poly = room_geom.room_polygon
    for entity in room_geom.entities.values():
        if entity.kind in ("wall", "room", "zone"):
            continue
        center = Point(entity.cx, entity.cy)
        if poly.contains(center) or poly.touches(center):
            graph.add_edge(entity.id, ROOM_NODE_ID, type="inside_room")
        else:
            graph.add_edge(entity.id, ROOM_NODE_ID, type="outside_room")


def _emit_object_to_object_edges(
    graph: nx.MultiDiGraph,
    room_geom: RoomGeometry,
    cfg: GraphConfig,
) -> None:
    objects = sorted(room_geom.furniture(), key=lambda e: e.id)
    n = len(objects)
    for i in range(n):
        src = objects[i]
        for j in range(n):
            if i == j:
                continue
            dst = objects[j]
            feats = _pair_features(src, dst, room_geom)
            d = feats["distance_m"]

            if d <= cfg.near_distance_m:
                graph.add_edge(
                    src.id,
                    dst.id,
                    type="near",
                    distance_m=d,
                )
            if (
                feats["facing_score"] > 0
                and angle_between_deg(
                    src.forward_axis_world(),
                    (dst.cx - src.cx, dst.cy - src.cy),
                )
                <= cfg.facing_angle_tol_deg
                and d <= cfg.far_distance_m * 1.5
            ):
                graph.add_edge(
                    src.id,
                    dst.id,
                    type="faces",
                    distance_m=d,
                    angle_error_deg=angle_between_deg(
                        src.forward_axis_world(),
                        (dst.cx - src.cx, dst.cy - src.cy),
                    ),
                )
            if feats["footprint_overlap_m2"] > cfg.overlap_eps_m2:
                graph.add_edge(
                    src.id,
                    dst.id,
                    type="overlaps",
                    overlap_m2=feats["footprint_overlap_m2"],
                )
            if (
                feats["relative_angle_deg"] <= cfg.parallel_yaw_tol_deg
                and d <= cfg.far_distance_m
            ):
                graph.add_edge(
                    src.id,
                    dst.id,
                    type="parallel_to",
                    angle_diff_deg=feats["relative_angle_deg"],
                )

            # Direction edges only when reasonably close so the graph stays
            # focused on local relationships.
            if d <= cfg.far_distance_m:
                direction = relative_direction(src, dst)
                etype = {
                    "front": "in_front_of",
                    "back": "behind",
                    "left": "left_of",
                    "right": "right_of",
                }[direction]
                graph.add_edge(
                    src.id,
                    dst.id,
                    type=etype,
                    distance_m=d,
                )


def _emit_object_to_wall_edges(
    graph: nx.MultiDiGraph,
    room_geom: RoomGeometry,
    cfg: GraphConfig,
) -> None:
    walls = list(room_geom.walls.values())
    if not walls:
        return
    for entity in room_geom.furniture():
        if entity.is_soft() and entity.label.lower() not in ("painting", "mirror", "curtain"):
            # Carpets etc. shouldn't form wall-relations.
            continue
        for wall in walls:
            clearance = footprint_clearance_to_wall(entity, wall)
            yaw_diff = yaw_diff_modpi_deg(entity.yaw_rad, wall.yaw_rad())
            if clearance <= cfg.touching_wall_m:
                graph.add_edge(
                    entity.id,
                    wall.id,
                    type="touching_wall",
                    clearance_m=clearance,
                    yaw_diff_deg=yaw_diff,
                )
            elif clearance <= cfg.against_wall_m:
                graph.add_edge(
                    entity.id,
                    wall.id,
                    type="against_wall",
                    clearance_m=clearance,
                    yaw_diff_deg=yaw_diff,
                )

            if (
                yaw_diff <= cfg.parallel_yaw_tol_deg
                and clearance <= cfg.against_wall_m * 2.0
            ):
                graph.add_edge(
                    entity.id,
                    wall.id,
                    type="parallel_to_wall",
                    clearance_m=clearance,
                    yaw_diff_deg=yaw_diff,
                )

            # has_backing: back edge of object hugs this wall and most of the
            # back edge nearest-points fall on the wall.
            back_clear = back_edge_distance_to_wall(entity, wall)
            if back_clear <= cfg.has_backing_max_clearance_m:
                coverage = back_edge_overlap_fraction(entity, wall)
                if coverage >= cfg.has_backing_min_coverage:
                    graph.add_edge(
                        entity.id,
                        wall.id,
                        type="has_backing",
                        clearance_m=back_clear,
                        coverage=coverage,
                    )


def _emit_door_window_edges(graph: nx.MultiDiGraph, room_geom: RoomGeometry) -> None:
    """Doors and windows belong to a parent wall (`raw.wall_id`)."""

    for ent in room_geom.entities.values():
        if ent.kind not in ("door", "window"):
            continue
        wall_id = str(ent.raw.get("wall_id", ""))
        if wall_id and wall_id in room_geom.entities:
            etype = "door_in_wall" if ent.kind == "door" else "window_in_wall"
            graph.add_edge(ent.id, wall_id, type=etype)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build_scene_graph(
    room_geom: RoomGeometry, cfg: GraphConfig | None = None
) -> nx.MultiDiGraph:
    cfg = cfg or room_geom.config
    graph = nx.MultiDiGraph()
    _add_entity_nodes(graph, room_geom)
    _add_room_node(graph, room_geom)
    _emit_inside_room_edges(graph, room_geom)
    _emit_object_to_object_edges(graph, room_geom, cfg)
    _emit_object_to_wall_edges(graph, room_geom, cfg)
    _emit_door_window_edges(graph, room_geom)
    return graph
