"""Stage 3 — functional graph: clusters, traffic paths, focal points."""

from __future__ import annotations

import heapq
import math
from typing import Iterable

import networkx as nx
import numpy as np
from shapely.geometry import LineString, Point

from .config import (
    BED_LABELS,
    DESK_LABELS,
    FOCAL_LABEL_PRIORITY,
    LAMP_LABELS,
    NIGHTSTAND_LABELS,
    PRIMARY_SEAT_LABELS,
    SEAT_LABELS,
    TABLE_LABELS,
    GraphConfig,
)
from .geometry import EntityGeom, RoomGeometry, walkable_grid
from .scene_graph import ROOM_NODE_ID
from .graph_types import Zone


# ---------------------------------------------------------------------------
# Cluster discovery
# ---------------------------------------------------------------------------


def _connected_components_within_radius(
    items: list[EntityGeom], radius_m: float
) -> list[list[EntityGeom]]:
    n = len(items)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(items[i].cx - items[j].cx, items[i].cy - items[j].cy)
            if d <= radius_m:
                union(i, j)

    groups: dict[int, list[EntityGeom]] = {}
    for i, item in enumerate(items):
        root = find(i)
        groups.setdefault(root, []).append(item)
    return list(groups.values())


def _cluster_centroid(members: Iterable[EntityGeom]) -> tuple[float, float]:
    members = list(members)
    if not members:
        return (0.0, 0.0)
    cx = sum(m.cx for m in members) / len(members)
    cy = sum(m.cy for m in members) / len(members)
    return (cx, cy)


def _build_seating_zones(
    room_geom: RoomGeometry, cfg: GraphConfig
) -> list[Zone]:
    seats = [
        e
        for e in room_geom.furniture()
        if e.label.lower() in SEAT_LABELS
    ]
    components = _connected_components_within_radius(seats, cfg.seating_cluster_radius_m)
    zones: list[Zone] = []
    for idx, members in enumerate(components):
        if not members:
            continue
        centroid = _cluster_centroid(members)
        zones.append(
            Zone(
                id=f"seating_group_{idx}",
                type="seating_group",
                members=[m.id for m in members],
                centroid_xy=centroid,
                attrs={
                    "primary_seat_id": _pick_primary_seat(members),
                    "size": len(members),
                },
            )
        )
    return zones


def _pick_primary_seat(members: list[EntityGeom]) -> str | None:
    if not members:
        return None
    primaries = [m for m in members if m.label.lower() in PRIMARY_SEAT_LABELS]
    pool = primaries if primaries else members
    return max(pool, key=lambda m: m.width * m.depth).id


def _build_bed_zones(room_geom: RoomGeometry, cfg: GraphConfig) -> list[Zone]:
    beds = [e for e in room_geom.furniture() if e.label.lower() in BED_LABELS]
    nightstands = [
        e for e in room_geom.furniture() if e.label.lower() in NIGHTSTAND_LABELS
    ]
    lamps = [e for e in room_geom.furniture() if e.label.lower() in LAMP_LABELS]

    zones: list[Zone] = []
    for idx, bed in enumerate(beds):
        members = [bed.id]
        bed_poly = bed.footprint()
        bed_pad = bed_poly.buffer(cfg.bed_nightstand_max_dist_m)
        nearby_ns = [n for n in nightstands if bed_pad.intersects(n.footprint())]
        members.extend(n.id for n in nearby_ns)
        nearby_lamps = [
            la
            for la in lamps
            if bed_pad.intersects(la.footprint())
            or any(
                la.footprint().distance(n.footprint()) <= 0.4 for n in nearby_ns
            )
        ]
        members.extend(la.id for la in nearby_lamps)
        zones.append(
            Zone(
                id=f"bed_group_{idx}",
                type="bed_group",
                members=members,
                centroid_xy=(bed.cx, bed.cy),
                attrs={
                    "bed_id": bed.id,
                    "nightstand_ids": [n.id for n in nearby_ns],
                    "lamp_ids": [la.id for la in nearby_lamps],
                },
            )
        )
    return zones


def _build_desk_zones(room_geom: RoomGeometry, cfg: GraphConfig) -> list[Zone]:
    desks = [e for e in room_geom.furniture() if e.label.lower() in DESK_LABELS]
    chairs = [e for e in room_geom.furniture() if e.label.lower() in SEAT_LABELS]
    computers = [
        e for e in room_geom.furniture() if e.label.lower() in {"computer"}
    ]
    lamps = [e for e in room_geom.furniture() if e.label.lower() in LAMP_LABELS]

    zones: list[Zone] = []
    counter = 0
    for desk in desks:
        members = [desk.id]
        desk_pad = desk.footprint().buffer(cfg.desk_chair_max_dist_m)
        near_chairs = [c for c in chairs if desk_pad.intersects(c.footprint())]
        near_comps = [c for c in computers if desk_pad.intersects(c.footprint())]
        near_lamps = [la for la in lamps if desk_pad.intersects(la.footprint())]
        if not (near_chairs or near_comps):
            # A bare desk without a chair or computer is not really a "desk_group".
            continue
        members.extend(c.id for c in near_chairs)
        members.extend(c.id for c in near_comps)
        members.extend(la.id for la in near_lamps)
        zones.append(
            Zone(
                id=f"desk_group_{counter}",
                type="desk_group",
                members=members,
                centroid_xy=(desk.cx, desk.cy),
                attrs={
                    "desk_id": desk.id,
                    "chair_ids": [c.id for c in near_chairs],
                    "computer_ids": [c.id for c in near_comps],
                    "lamp_ids": [la.id for la in near_lamps],
                },
            )
        )
        counter += 1
    return zones


# ---------------------------------------------------------------------------
# Functional edges
# ---------------------------------------------------------------------------


def _attach_zone_nodes(graph: nx.MultiDiGraph, zones: list[Zone]) -> None:
    for zone in zones:
        graph.add_node(
            zone.id,
            type="zone",
            kind="zone",
            label=zone.type,
            geometry={"centroid_xy": list(zone.centroid_xy or (0.0, 0.0))},
            attrs={"members": list(zone.members), **zone.attrs},
        )
        for member in zone.members:
            if not graph.has_node(member):
                continue
            graph.add_edge(member, zone.id, type="participates_in")


def _emit_table_for_edges(
    graph: nx.MultiDiGraph,
    room_geom: RoomGeometry,
    seating_zones: list[Zone],
    cfg: GraphConfig,
) -> None:
    tables = [e for e in room_geom.furniture() if e.label.lower() in TABLE_LABELS]
    for table in tables:
        for zone in seating_zones:
            if not zone.members:
                continue
            cx, cy = zone.centroid_xy or (0.0, 0.0)
            d = math.hypot(table.cx - cx, table.cy - cy)
            if d <= cfg.seating_cluster_radius_m + cfg.table_for_max_dist_m:
                graph.add_edge(table.id, zone.id, type="serves", distance_m=d)
                # Also connect to the primary seat for "table_for" specifically.
                primary = zone.attrs.get("primary_seat_id")
                if primary:
                    graph.add_edge(table.id, primary, type="table_for", distance_m=d)


# ---------------------------------------------------------------------------
# Walkable grid + A* for traffic paths
# ---------------------------------------------------------------------------


def _world_to_cell(
    pt: tuple[float, float],
    bounds: tuple[float, float, float, float],
    cell_m: float,
    grid_shape: tuple[int, int],
) -> tuple[int, int]:
    min_x, min_y, _, _ = bounds
    rows, cols = grid_shape
    c = int((pt[0] - min_x) / cell_m)
    r = int((pt[1] - min_y) / cell_m)
    c = max(0, min(cols - 1, c))
    r = max(0, min(rows - 1, r))
    return (r, c)


def _cell_to_world(
    cell: tuple[int, int],
    bounds: tuple[float, float, float, float],
    cell_m: float,
) -> tuple[float, float]:
    r, c = cell
    min_x, min_y, _, _ = bounds
    return (min_x + (c + 0.5) * cell_m, min_y + (r + 0.5) * cell_m)


def _nearest_walkable(
    grid: np.ndarray, cell: tuple[int, int]
) -> tuple[int, int] | None:
    rows, cols = grid.shape
    if grid[cell]:
        return cell
    best: tuple[int, int] | None = None
    best_d = math.inf
    for r in range(rows):
        for c in range(cols):
            if not grid[r, c]:
                continue
            d = (r - cell[0]) ** 2 + (c - cell[1]) ** 2
            if d < best_d:
                best_d = d
                best = (r, c)
    return best


def _astar(
    grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]
) -> list[tuple[int, int]] | None:
    rows, cols = grid.shape
    if not grid[start] or not grid[goal]:
        return None

    def h(a: tuple[int, int], b: tuple[int, int]) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    open_heap: list[tuple[float, tuple[int, int]]] = []
    heapq.heappush(open_heap, (h(start, goal), start))
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], float] = {start: 0.0}

    neighbors = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
        (-1, -1),
        (-1, 1),
        (1, -1),
        (1, 1),
    ]

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        for dr, dc in neighbors:
            nr, nc = current[0] + dr, current[1] + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            if not grid[nr, nc]:
                continue
            step = math.hypot(dr, dc)
            tentative = g_score[current] + step
            if tentative < g_score.get((nr, nc), math.inf):
                came_from[(nr, nc)] = current
                g_score[(nr, nc)] = tentative
                f_score = tentative + h((nr, nc), goal)
                heapq.heappush(open_heap, (f_score, (nr, nc)))
    return None


def _path_corridor_widths(
    grid: np.ndarray, path: list[tuple[int, int]], cell_m: float
) -> list[float]:
    """Approximate the corridor half-width at each cell along the path.

    Uses a small BFS bounded to ~1.5 m radius so the cost stays small.
    """

    rows, cols = grid.shape
    max_radius_cells = int(math.ceil(1.5 / cell_m))
    widths: list[float] = []
    for r0, c0 in path:
        # BFS outward to find nearest non-walkable cell.
        from collections import deque

        seen = {(r0, c0)}
        queue: deque[tuple[int, int, int]] = deque([(r0, c0, 0)])
        nearest: int | None = None
        while queue:
            r, c, depth = queue.popleft()
            if depth > max_radius_cells:
                break
            if not grid[r, c]:
                nearest = depth
                break
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        nearest = depth + 1
                        continue
                    if (nr, nc) in seen:
                        continue
                    seen.add((nr, nc))
                    queue.append((nr, nc, depth + 1))
            if nearest is not None:
                break
        widths.append((nearest if nearest is not None else max_radius_cells) * cell_m)
    return widths


def _emit_traffic_edges(
    graph: nx.MultiDiGraph,
    room_geom: RoomGeometry,
    zones: list[Zone],
    cfg: GraphConfig,
) -> None:
    if not room_geom.entry_door_ids:
        return

    grid, bounds = walkable_grid(room_geom.walkable_polygon, cfg.walkable_grid_cell_m)
    if grid.size == 0 or not grid.any():
        return

    # Snap door and zone centroids to nearest walkable cell.
    targets: list[tuple[str, tuple[float, float]]] = []
    for zone in zones:
        if zone.centroid_xy is None:
            continue
        if zone.type == "seating_group":
            primary = zone.attrs.get("primary_seat_id")
            if primary and primary in room_geom.entities:
                ent = room_geom.entities[primary]
                targets.append((zone.id, (ent.cx, ent.cy)))
                continue
        if zone.type == "bed_group":
            bed_id = zone.attrs.get("bed_id")
            if bed_id and bed_id in room_geom.entities:
                bed = room_geom.entities[bed_id]
                targets.append((zone.id, (bed.cx, bed.cy)))
                continue
        targets.append((zone.id, zone.centroid_xy))

    for door_id in room_geom.entry_door_ids:
        door = room_geom.entities.get(door_id)
        if door is None:
            continue
        door_cell = _world_to_cell(door.center, bounds, cfg.walkable_grid_cell_m, grid.shape)
        door_walkable = _nearest_walkable(grid, door_cell)
        if door_walkable is None:
            continue

        for zone_id, target_xy in targets:
            target_cell = _world_to_cell(target_xy, bounds, cfg.walkable_grid_cell_m, grid.shape)
            target_walkable = _nearest_walkable(grid, target_cell)
            if target_walkable is None:
                continue
            path = _astar(grid, door_walkable, target_walkable)
            if not path:
                continue

            world_path = [
                _cell_to_world(cell, bounds, cfg.walkable_grid_cell_m) for cell in path
            ]
            length_m = sum(
                math.hypot(world_path[i + 1][0] - world_path[i][0], world_path[i + 1][1] - world_path[i][1])
                for i in range(len(world_path) - 1)
            )
            graph.add_edge(
                door_id,
                zone_id,
                type="traffic_path",
                length_m=length_m,
                cells=len(path),
            )

            # Detect blocking furniture: cells where corridor is too tight.
            widths = _path_corridor_widths(grid, path, cfg.walkable_grid_cell_m)
            tight_world_points = [
                world_path[i]
                for i, w in enumerate(widths)
                if w < cfg.corridor_min_width_m
            ]
            if tight_world_points:
                _mark_blockers(graph, room_geom, door_id, zone_id, tight_world_points, cfg)


def _mark_blockers(
    graph: nx.MultiDiGraph,
    room_geom: RoomGeometry,
    door_id: str,
    zone_id: str,
    tight_points: list[tuple[float, float]],
    cfg: GraphConfig,
) -> None:
    if not tight_points:
        return
    candidates = [
        e for e in room_geom.furniture() if not e.is_soft() and e.height >= 0.2
    ]
    seen: set[str] = set()
    for px, py in tight_points:
        pt = Point(px, py)
        # Find any furniture within (corridor_min_width / 2) of this tight point.
        for ent in candidates:
            d = ent.footprint().distance(pt)
            if d <= cfg.corridor_min_width_m / 2.0 + cfg.walkable_grid_cell_m:
                key = ent.id
                if key in seen:
                    continue
                seen.add(key)
                graph.add_edge(
                    ent.id,
                    zone_id,
                    type="blocks_path",
                    from_door=door_id,
                    distance_m=d,
                )


def _emit_entry_path_edges(
    graph: nx.MultiDiGraph, room_geom: RoomGeometry, zones: list[Zone]
) -> None:
    """Add a higher-level `entry_path` edge from each entry door to the primary
    occupant of every seating/bed zone."""

    for zone in zones:
        if zone.type not in ("seating_group", "bed_group"):
            continue
        target_id: str | None = None
        if zone.type == "seating_group":
            target_id = zone.attrs.get("primary_seat_id")
        elif zone.type == "bed_group":
            target_id = zone.attrs.get("bed_id")
        if not target_id:
            continue
        for door_id in room_geom.entry_door_ids:
            graph.add_edge(door_id, target_id, type="entry_path")


# ---------------------------------------------------------------------------
# Focal point
# ---------------------------------------------------------------------------


def _pick_focal_point(
    room_geom: RoomGeometry, zones: list[Zone]
) -> str | None:
    label_lookup: dict[str, list[EntityGeom]] = {}
    for ent in room_geom.furniture():
        label_lookup.setdefault(ent.label.lower(), []).append(ent)

    for label in FOCAL_LABEL_PRIORITY:
        if label in label_lookup and label_lookup[label]:
            return max(label_lookup[label], key=lambda e: e.width * e.depth).id

    windows = list(room_geom.windows())
    if windows:
        return max(windows, key=lambda e: e.width * e.height).id

    seating_zones = [z for z in zones if z.type == "seating_group"]
    if seating_zones:
        primary = seating_zones[0].attrs.get("primary_seat_id")
        if primary:
            return primary

    return None


def _emit_focal_edge(
    graph: nx.MultiDiGraph, focal_id: str | None
) -> None:
    if focal_id is None:
        return
    graph.add_edge(focal_id, ROOM_NODE_ID, type="focal_point_of")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def apply_functional_layer(
    graph: nx.MultiDiGraph,
    room_geom: RoomGeometry,
    cfg: GraphConfig | None = None,
) -> tuple[list[Zone], str | None]:
    cfg = cfg or room_geom.config

    seating_zones = _build_seating_zones(room_geom, cfg)
    bed_zones = _build_bed_zones(room_geom, cfg)
    desk_zones = _build_desk_zones(room_geom, cfg)
    zones = seating_zones + bed_zones + desk_zones

    _attach_zone_nodes(graph, zones)
    _emit_table_for_edges(graph, room_geom, seating_zones, cfg)
    _emit_traffic_edges(graph, room_geom, zones, cfg)
    _emit_entry_path_edges(graph, room_geom, zones)

    focal_id = _pick_focal_point(room_geom, zones)
    _emit_focal_edge(graph, focal_id)

    return zones, focal_id
