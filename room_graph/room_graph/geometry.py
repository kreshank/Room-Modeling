"""Geometric pre-pass for SpatialLM scene.json entities.

Top-down (XY plane) only. All higher-level stages (scene graph, functional,
fengshui) read derived geometry from here so that no other module recomputes
polygons, oriented rectangles, or line-of-sight checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np
from shapely.geometry import LineString, MultiLineString, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import polygonize, unary_union

from .config import (
    LOW_HEIGHT_LABELS,
    SOFT_LABELS,
    GraphConfig,
    forward_axis_for_label,
)


# ---------------------------------------------------------------------------
# Per-entity geometry
# ---------------------------------------------------------------------------


@dataclass
class WallSegment:
    """An axis-of-line representation of a wall (zero-thickness segment)."""

    id: str
    ax: float
    ay: float
    bx: float
    by: float
    height: float
    thickness: float

    def line(self) -> LineString:
        return LineString([(self.ax, self.ay), (self.bx, self.by)])

    def length(self) -> float:
        return math.hypot(self.bx - self.ax, self.by - self.ay)

    def yaw_rad(self) -> float:
        return math.atan2(self.by - self.ay, self.bx - self.ax)

    def tangent(self) -> tuple[float, float]:
        L = self.length() or 1.0
        return ((self.bx - self.ax) / L, (self.by - self.ay) / L)

    def normals(self) -> tuple[tuple[float, float], tuple[float, float]]:
        tx, ty = self.tangent()
        return ((ty, -tx), (-ty, tx))

    def midpoint(self) -> tuple[float, float]:
        return (0.5 * (self.ax + self.bx), 0.5 * (self.ay + self.by))


@dataclass
class EntityGeom:
    """Top-down oriented rectangle plus derived shapely geometry for one entity."""

    id: str
    kind: str
    label: str
    cx: float
    cy: float
    z: float
    width: float
    depth: float
    height: float
    yaw_rad: float

    raw: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._footprint_cache: Polygon | None = None

    @property
    def center(self) -> tuple[float, float]:
        return (self.cx, self.cy)

    @property
    def half_size(self) -> tuple[float, float]:
        return (0.5 * self.width, 0.5 * self.depth)

    def yaw_deg(self) -> float:
        return math.degrees(self.yaw_rad)

    def forward_axis_world(self) -> tuple[float, float]:
        lx, ly = forward_axis_for_label(self.label)
        c, s = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        return (lx * c - ly * s, lx * s + ly * c)

    def right_axis_world(self) -> tuple[float, float]:
        fx, fy = self.forward_axis_world()
        return (fy, -fx)

    def corners(self) -> list[tuple[float, float]]:
        """Top-down corner points in CCW order: BL, BR, TR, TL of local rect."""
        hw, hd = self.half_size
        local = [(-hw, -hd), (hw, -hd), (hw, hd), (-hw, hd)]
        c, s = math.cos(self.yaw_rad), math.sin(self.yaw_rad)
        return [
            (self.cx + lx * c - ly * s, self.cy + lx * s + ly * c) for (lx, ly) in local
        ]

    def footprint(self) -> Polygon:
        if self._footprint_cache is not None:
            return self._footprint_cache
        poly = Polygon(self.corners())
        if not poly.is_valid:
            poly = poly.buffer(0)
        self._footprint_cache = poly
        return poly

    def back_edge(self) -> LineString:
        """Top-down line segment along the entity's back face (opposite forward)."""
        corners = self.corners()
        # Back edge in local frame: from (-hw, -hd) to (hw, -hd) when forward=+y.
        # Our `corners()` returns BL, BR, TR, TL in that local frame, so back is BL->BR.
        # If a label uses a different forward axis, we still treat "back" as opposite
        # of forward in world space.
        fx, fy = self.forward_axis_world()
        # Compute back-edge by selecting the rectangle edge whose midpoint is farthest
        # along -forward direction.
        edges = [
            (corners[0], corners[1]),
            (corners[1], corners[2]),
            (corners[2], corners[3]),
            (corners[3], corners[0]),
        ]
        best = max(
            edges,
            key=lambda e: -(
                (0.5 * (e[0][0] + e[1][0]) - self.cx) * fx
                + (0.5 * (e[0][1] + e[1][1]) - self.cy) * fy
            ),
        )
        return LineString([best[0], best[1]])

    def is_soft(self) -> bool:
        return self.label.lower() in SOFT_LABELS

    def is_low(self) -> bool:
        return self.label.lower() in LOW_HEIGHT_LABELS

    def to_dict(self) -> dict[str, Any]:
        return {
            "cx": self.cx,
            "cy": self.cy,
            "z": self.z,
            "width": self.width,
            "depth": self.depth,
            "height": self.height,
            "yaw_rad": self.yaw_rad,
            "yaw_deg": self.yaw_deg(),
            "forward_axis_world": list(self.forward_axis_world()),
            "corners_xy": [list(c) for c in self.corners()],
        }


# ---------------------------------------------------------------------------
# Top-level RoomGeometry container
# ---------------------------------------------------------------------------


@dataclass
class RoomGeometry:
    config: GraphConfig
    entities: dict[str, EntityGeom]
    walls: dict[str, WallSegment]

    room_polygon: Polygon
    walkable_polygon: BaseGeometry
    occluders: BaseGeometry

    entry_door_ids: list[str]
    passage_door_ids: list[str]

    def furniture(self) -> list[EntityGeom]:
        return [e for e in self.entities.values() if e.kind in ("object", "furniture")]

    def doors(self) -> list[EntityGeom]:
        return [e for e in self.entities.values() if e.kind == "door"]

    def windows(self) -> list[EntityGeom]:
        return [e for e in self.entities.values() if e.kind == "window"]

    def line_of_sight(self, a: tuple[float, float], b: tuple[float, float]) -> bool:
        if a == b:
            return True
        seg = LineString([a, b])
        if self.occluders.is_empty:
            return True
        # Allow tiny epsilon overlap with endpoints.
        inter = seg.intersection(self.occluders)
        if inter.is_empty:
            return True
        # If the only intersection points are the endpoints themselves, count as visible.
        if inter.geom_type == "Point":
            d_a = inter.distance(Point(a))
            d_b = inter.distance(Point(b))
            return min(d_a, d_b) < 1e-3
        return False


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------


def _snap(v: float, step: float) -> float:
    return round(v / step) * step


def _snapped_segment(
    ax: float, ay: float, bx: float, by: float, step: float
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    a = (_snap(ax, step), _snap(ay, step))
    b = (_snap(bx, step), _snap(by, step))
    if a == b:
        return None
    return (a, b)


def _entity_from_record(record: dict[str, Any]) -> EntityGeom:
    return EntityGeom(
        id=record["id"],
        kind=str(record.get("kind", "unknown")),
        label=str(record.get("label", record.get("kind", ""))),
        cx=float(record["x"]),
        cy=float(record["y"]),
        z=float(record.get("z", 0.0)),
        width=float(record.get("width", 0.0)),
        depth=float(record.get("depth", 0.0)),
        height=float(record.get("height", 0.0)),
        yaw_rad=float(record.get("yaw_rad", 0.0)),
        raw=dict(record.get("raw", {})),
    )


def _wall_from_record(record: dict[str, Any]) -> WallSegment | None:
    raw = record.get("raw") or {}
    try:
        ax = float(raw["ax"])
        ay = float(raw["ay"])
        bx = float(raw["bx"])
        by = float(raw["by"])
    except (KeyError, TypeError, ValueError):
        return None
    return WallSegment(
        id=record["id"],
        ax=ax,
        ay=ay,
        bx=bx,
        by=by,
        height=float(record.get("height", raw.get("height", 0.0))),
        thickness=float(raw.get("thickness", 0.0)),
    )


def reconstruct_room_polygon(
    walls: Iterable[WallSegment], snap_m: float
) -> Polygon | None:
    """Build a closed top-down polygon from wall segments.

    Snap endpoints to a fine grid, union, polygonize, and pick the largest
    polygon by area. Falls back to convex hull of wall endpoints when no
    closed loop is available.
    """

    segments: list[LineString] = []
    points: list[tuple[float, float]] = []
    for w in walls:
        snapped = _snapped_segment(w.ax, w.ay, w.bx, w.by, snap_m)
        if snapped is None:
            continue
        a, b = snapped
        segments.append(LineString([a, b]))
        points.extend([a, b])

    if not segments:
        return None

    merged = unary_union(MultiLineString(segments))
    polys = list(polygonize(merged))
    if polys:
        return max(polys, key=lambda p: p.area)

    # Fallback: convex hull of endpoints if it has positive area.
    if len(points) >= 3:
        hull = Polygon(points).convex_hull
        if isinstance(hull, Polygon) and hull.area > 0:
            return hull
    return None


def _is_blocker(entity: EntityGeom) -> bool:
    """Return True if `entity` blocks walking on the floor.

    Soft items like carpets and curtains do not. Low items like stools and
    side tables still block walking. Tall opaque items block both walking and
    line-of-sight.
    """

    label = entity.label.lower()
    if entity.kind not in ("object", "furniture"):
        return False
    if label in SOFT_LABELS:
        return False
    if entity.height < 0.2:
        return False
    return True


def _is_tall_occluder(entity: EntityGeom, min_height: float) -> bool:
    if entity.kind not in ("object", "furniture"):
        return False
    if entity.label.lower() in SOFT_LABELS:
        return False
    if entity.height < min_height:
        return False
    return True


def compute_walkable_polygon(
    room_polygon: Polygon, entities: Iterable[EntityGeom]
) -> BaseGeometry:
    blockers = [e.footprint() for e in entities if _is_blocker(e)]
    if not blockers:
        return room_polygon
    blocked = unary_union(blockers)
    walkable = room_polygon.difference(blocked)
    if walkable.is_empty:
        return room_polygon
    return walkable


def compute_occluders(
    walls: Iterable[WallSegment],
    entities: Iterable[EntityGeom],
    min_tall_height: float,
) -> BaseGeometry:
    """Geometry treated as opaque to line-of-sight at standing eye height."""

    parts: list[BaseGeometry] = []
    for w in walls:
        if w.length() <= 0:
            continue
        parts.append(w.line())
    for e in entities:
        if _is_tall_occluder(e, min_tall_height):
            parts.append(e.footprint())
    if not parts:
        return MultiLineString([])
    return unary_union(parts)


def classify_doors(
    doors: Iterable[EntityGeom],
    walls: dict[str, WallSegment],
    room_polygon: Polygon,
) -> tuple[list[str], list[str]]:
    """Split doors into entry doors (open to outside) and passage doors.

    A door's outward normal is taken from its parent wall. A small probe point
    on each side of the door's center decides whether that side is inside the
    room polygon.
    """

    entries: list[str] = []
    passages: list[str] = []
    probe = 0.20

    for door in doors:
        wall_id = str(door.raw.get("wall_id", ""))
        wall = walls.get(wall_id)
        if wall is None:
            tx = math.cos(door.yaw_rad)
            ty = math.sin(door.yaw_rad)
        else:
            tx, ty = wall.tangent()
        nx, ny = ty, -tx
        cx, cy = door.center
        a = Point(cx + nx * probe, cy + ny * probe)
        b = Point(cx - nx * probe, cy - ny * probe)
        a_in = room_polygon.contains(a) or room_polygon.touches(a)
        b_in = room_polygon.contains(b) or room_polygon.touches(b)
        if a_in != b_in:
            entries.append(door.id)
        elif a_in and b_in:
            passages.append(door.id)
        else:
            entries.append(door.id)
    return entries, passages


def build_room_geometry(
    scene: dict[str, Any], config: GraphConfig | None = None
) -> RoomGeometry:
    cfg = config or GraphConfig()

    entities: dict[str, EntityGeom] = {}
    walls: dict[str, WallSegment] = {}
    for record in scene.get("entities", []):
        ent = _entity_from_record(record)
        entities[ent.id] = ent
        if ent.kind == "wall":
            wall = _wall_from_record(record)
            if wall is not None:
                walls[wall.id] = wall

    room_polygon = reconstruct_room_polygon(walls.values(), cfg.grid_snap_m)
    if room_polygon is None or room_polygon.area <= 0:
        room_polygon = _fallback_polygon_from_entities(entities.values())

    walkable = compute_walkable_polygon(room_polygon, entities.values())
    occluders = compute_occluders(walls.values(), entities.values(), cfg.los_opaque_min_height_m)

    door_geoms = [e for e in entities.values() if e.kind == "door"]
    entries, passages = classify_doors(door_geoms, walls, room_polygon)

    return RoomGeometry(
        config=cfg,
        entities=entities,
        walls=walls,
        room_polygon=room_polygon,
        walkable_polygon=walkable,
        occluders=occluders,
        entry_door_ids=entries,
        passage_door_ids=passages,
    )


def _fallback_polygon_from_entities(entities: Iterable[EntityGeom]) -> Polygon:
    """Last-resort bounding rectangle when wall reconstruction fails."""

    xs: list[float] = []
    ys: list[float] = []
    for e in entities:
        for cx, cy in e.corners():
            xs.append(cx)
            ys.append(cy)
    if not xs or not ys:
        return Polygon([(-1, -1), (1, -1), (1, 1), (-1, 1)])
    return Polygon(
        [
            (min(xs), min(ys)),
            (max(xs), min(ys)),
            (max(xs), max(ys)),
            (min(xs), max(ys)),
        ]
    )


# ---------------------------------------------------------------------------
# Public utility helpers used across stages
# ---------------------------------------------------------------------------


def angle_between_deg(u: tuple[float, float], v: tuple[float, float]) -> float:
    """Unsigned angle between two 2D vectors, in degrees, in [0, 180]."""

    nu = math.hypot(*u)
    nv = math.hypot(*v)
    if nu == 0 or nv == 0:
        return 0.0
    cosv = max(-1.0, min(1.0, (u[0] * v[0] + u[1] * v[1]) / (nu * nv)))
    return math.degrees(math.acos(cosv))


def signed_yaw_diff_deg(a_rad: float, b_rad: float) -> float:
    """Signed difference (a - b) wrapped to [-180, 180]."""

    d = math.degrees(a_rad - b_rad)
    while d > 180.0:
        d -= 360.0
    while d < -180.0:
        d += 360.0
    return d


def yaw_diff_modpi_deg(a_rad: float, b_rad: float) -> float:
    """Unsigned yaw difference modulo pi (so 0deg and 180deg are equivalent)."""

    d = abs(signed_yaw_diff_deg(a_rad, b_rad))
    return min(d, 180.0 - d)


def relative_direction(source: EntityGeom, target: EntityGeom) -> str:
    """Return one of front/back/left/right relative to source's local frame."""

    fx, fy = source.forward_axis_world()
    rx, ry = source.right_axis_world()
    dx = target.cx - source.cx
    dy = target.cy - source.cy
    forward = dx * fx + dy * fy
    right = dx * rx + dy * ry
    if abs(forward) >= abs(right):
        return "front" if forward >= 0 else "back"
    return "right" if right >= 0 else "left"


def footprint_clearance_to_wall(entity: EntityGeom, wall: WallSegment) -> float:
    return entity.footprint().distance(wall.line())


def back_edge_distance_to_wall(entity: EntityGeom, wall: WallSegment) -> float:
    return entity.back_edge().distance(wall.line())


def back_edge_overlap_fraction(entity: EntityGeom, wall: WallSegment) -> float:
    """Return the fraction of the entity's back edge whose nearest wall point lies on `wall`.

    Approximated by sampling 9 points along the back edge and measuring distances.
    """

    line = wall.line()
    back = entity.back_edge()
    n = 9
    if back.length <= 0:
        return 0.0
    hits = 0
    for i in range(n):
        s = i / (n - 1)
        p = back.interpolate(s, normalized=True)
        if line.distance(p) <= 0.20:
            hits += 1
    return hits / n


def project_to_world(rect: EntityGeom, local: tuple[float, float]) -> tuple[float, float]:
    c, s = math.cos(rect.yaw_rad), math.sin(rect.yaw_rad)
    return (rect.cx + local[0] * c - local[1] * s, rect.cy + local[0] * s + local[1] * c)


def polygon_to_xy(geom: BaseGeometry) -> list[list[list[float]]]:
    """Flatten any (Multi)Polygon into a list of exterior rings as [[x, y], ...]."""

    rings: list[list[list[float]]] = []
    if geom.is_empty:
        return rings
    if isinstance(geom, Polygon):
        rings.append([list(p) for p in geom.exterior.coords])
    elif isinstance(geom, MultiPolygon):
        for poly in geom.geoms:
            rings.append([list(p) for p in poly.exterior.coords])
    return rings


def walkable_grid(
    walkable: BaseGeometry, cell_m: float
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Rasterize a walkable polygon to a boolean occupancy grid.

    Returns `(grid, (min_x, min_y, max_x, max_y))`. `grid[r, c]` is True when
    the cell center is walkable.
    """

    if walkable.is_empty:
        return np.zeros((1, 1), dtype=bool), (0.0, 0.0, cell_m, cell_m)
    min_x, min_y, max_x, max_y = walkable.bounds
    cols = max(1, int(math.ceil((max_x - min_x) / cell_m)))
    rows = max(1, int(math.ceil((max_y - min_y) / cell_m)))
    grid = np.zeros((rows, cols), dtype=bool)
    for r in range(rows):
        cy = min_y + (r + 0.5) * cell_m
        for c in range(cols):
            cx = min_x + (c + 0.5) * cell_m
            grid[r, c] = walkable.contains(Point(cx, cy))
    return grid, (min_x, min_y, max_x, max_y)
