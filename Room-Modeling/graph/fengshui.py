"""Stage 4 — feng shui principle layer.

Each rule consumes the geometric + functional graph and emits structured
`PrincipleCheck` records. No numeric scoring is combined here — that is the
scorer's job. Each rule is responsible for explaining its evidence.
"""

from __future__ import annotations

import math
from typing import Iterable

import networkx as nx
from shapely.geometry import LineString, Point

from .config import (
    BED_LABELS,
    DESK_LABELS,
    LAMP_LABELS,
    NIGHTSTAND_LABELS,
    PRIMARY_SEAT_LABELS,
    GraphConfig,
)
from .geometry import (
    EntityGeom,
    RoomGeometry,
    WallSegment,
    angle_between_deg,
    yaw_diff_modpi_deg,
)
from .graph_types import PrincipleCheck, Zone


# ---------------------------------------------------------------------------
# Graph query helpers
# ---------------------------------------------------------------------------


def _outgoing_edges_of_type(
    graph: nx.MultiDiGraph, source: str, edge_type: str
) -> list[tuple[str, str, dict]]:
    if source not in graph:
        return []
    out: list[tuple[str, str, dict]] = []
    for _, dst, data in graph.out_edges(source, data=True):
        if data.get("type") == edge_type:
            out.append((source, dst, data))
    return out


def _has_backing_walls(
    graph: nx.MultiDiGraph, source: str
) -> list[str]:
    return [dst for _, dst, _ in _outgoing_edges_of_type(graph, source, "has_backing")]


def _faces_door(
    entity: EntityGeom, door: EntityGeom, room_geom: RoomGeometry
) -> tuple[bool, float]:
    """Return (visible_via_los, angle_error_deg) between entity forward axis and door center."""

    visible = room_geom.line_of_sight(entity.center, door.center)
    fwd = entity.forward_axis_world()
    vec = (door.cx - entity.cx, door.cy - entity.cy)
    err = angle_between_deg(fwd, vec)
    return visible, err


# ---------------------------------------------------------------------------
# Principles
# ---------------------------------------------------------------------------


def _principle_command_position(
    graph: nx.MultiDiGraph,
    room_geom: RoomGeometry,
    zones: list[Zone],
    cfg: GraphConfig,
) -> list[PrincipleCheck]:
    """A "main" piece of furniture should: see the door, sit far enough back, not
    be in the door's direct path, and have a wall behind it."""

    candidates: list[EntityGeom] = []
    seen: set[str] = set()

    for zone in zones:
        if zone.type == "seating_group":
            seat_id = zone.attrs.get("primary_seat_id")
            if seat_id and seat_id not in seen:
                ent = room_geom.entities.get(seat_id)
                if ent and ent.label.lower() in PRIMARY_SEAT_LABELS:
                    candidates.append(ent)
                    seen.add(seat_id)
        if zone.type == "bed_group":
            bed_id = zone.attrs.get("bed_id")
            if bed_id and bed_id not in seen:
                ent = room_geom.entities.get(bed_id)
                if ent:
                    candidates.append(ent)
                    seen.add(bed_id)
        if zone.type == "desk_group":
            desk_id = zone.attrs.get("desk_id")
            if desk_id and desk_id not in seen:
                ent = room_geom.entities.get(desk_id)
                if ent:
                    candidates.append(ent)
                    seen.add(desk_id)

    checks: list[PrincipleCheck] = []
    if not room_geom.entry_door_ids:
        return checks

    for entity in candidates:
        evidence: list[str] = []
        score = 1.0
        violations = 0
        warnings = 0

        # Use the closest entry door for the primary check.
        door = min(
            (room_geom.entities[d] for d in room_geom.entry_door_ids if d in room_geom.entities),
            key=lambda d: math.hypot(d.cx - entity.cx, d.cy - entity.cy),
            default=None,
        )
        if door is None:
            continue

        visible, angle_err = _faces_door(entity, door, room_geom)
        d_door = math.hypot(door.cx - entity.cx, door.cy - entity.cy)

        if visible:
            evidence.append(f"entry door {door.id} is visible from {entity.id}")
        else:
            evidence.append(f"entry door {door.id} is NOT visible from {entity.id}")
            score -= 0.4
            violations += 1

        if d_door < cfg.command_min_door_dist_m:
            evidence.append(
                f"too close to entry: {d_door:.2f}m < {cfg.command_min_door_dist_m:.2f}m"
            )
            score -= 0.2
            warnings += 1
        else:
            evidence.append(f"distance to entry door {d_door:.2f}m is comfortable")

        # Aligned with door = bad: door faces directly into the entity.
        wall_id = str(door.raw.get("wall_id", ""))
        wall = room_geom.walls.get(wall_id)
        if wall is not None:
            tx, ty = wall.tangent()
            nx_, ny_ = ty, -tx
            # Long line through door perpendicular to wall.
            ray = LineString(
                [
                    (door.cx + nx_ * 30.0, door.cy + ny_ * 30.0),
                    (door.cx - nx_ * 30.0, door.cy - ny_ * 30.0),
                ]
            )
            if ray.intersects(entity.footprint()):
                evidence.append(
                    f"directly aligned with door {door.id} (door axis crosses footprint)"
                )
                score -= 0.3
                violations += 1
            else:
                evidence.append(f"not directly aligned with door {door.id}")

        backings = _has_backing_walls(graph, entity.id)
        if backings:
            evidence.append(f"has wall backing ({', '.join(backings)})")
        else:
            evidence.append("lacks wall backing")
            score -= 0.3
            warnings += 1

        score = max(0.0, min(1.0, score))
        if violations > 0:
            status = "violated"
        elif warnings > 0:
            status = "weak"
        else:
            status = "good"

        checks.append(
            PrincipleCheck(
                principle="command_position",
                target=entity.id,
                status=status,
                score=score,
                evidence=evidence,
                edges=[
                    {"source": entity.id, "target": door.id, "type": "evaluated_against_door"},
                ],
            )
        )

    return checks


def _principle_solid_backing(
    graph: nx.MultiDiGraph,
    room_geom: RoomGeometry,
    zones: list[Zone],
    cfg: GraphConfig,
) -> list[PrincipleCheck]:
    """Bed and primary seating should have a wall behind them with no door/window
    cut over the back footprint."""

    targets: list[EntityGeom] = []
    seen: set[str] = set()
    for zone in zones:
        if zone.type == "bed_group":
            bid = zone.attrs.get("bed_id")
            if bid and bid in room_geom.entities and bid not in seen:
                targets.append(room_geom.entities[bid])
                seen.add(bid)
        if zone.type == "seating_group":
            sid = zone.attrs.get("primary_seat_id")
            if (
                sid
                and sid in room_geom.entities
                and sid not in seen
                and room_geom.entities[sid].label.lower() in PRIMARY_SEAT_LABELS
            ):
                targets.append(room_geom.entities[sid])
                seen.add(sid)

    fixtures = [
        e for e in room_geom.entities.values() if e.kind in ("door", "window")
    ]

    checks: list[PrincipleCheck] = []
    for entity in targets:
        evidence: list[str] = []
        backings = _has_backing_walls(graph, entity.id)
        if not backings:
            checks.append(
                PrincipleCheck(
                    principle="solid_backing",
                    target=entity.id,
                    status="violated",
                    score=0.0,
                    evidence=[f"{entity.id} has no wall backing"],
                )
            )
            continue

        # Check whether back edge crosses any door/window.
        back_edge = entity.back_edge()
        cut_by: list[str] = []
        for fix in fixtures:
            if back_edge.distance(fix.footprint()) < 0.2 and back_edge.intersects(
                fix.footprint().buffer(0.05)
            ):
                cut_by.append(fix.id)

        if cut_by:
            evidence.append(f"back of {entity.id} is cut by {', '.join(cut_by)}")
            checks.append(
                PrincipleCheck(
                    principle="solid_backing",
                    target=entity.id,
                    status="weak",
                    score=0.4,
                    evidence=evidence,
                )
            )
        else:
            evidence.append(
                f"{entity.id} has solid wall backing ({', '.join(backings)})"
            )
            checks.append(
                PrincipleCheck(
                    principle="solid_backing",
                    target=entity.id,
                    status="good",
                    score=1.0,
                    evidence=evidence,
                )
            )

    return checks


def _principle_bed_aligned_with_door(
    graph: nx.MultiDiGraph,
    room_geom: RoomGeometry,
    zones: list[Zone],
    cfg: GraphConfig,
) -> list[PrincipleCheck]:
    beds = [
        room_geom.entities[z.attrs["bed_id"]]
        for z in zones
        if z.type == "bed_group" and z.attrs.get("bed_id") in room_geom.entities
    ]
    doors = [e for e in room_geom.entities.values() if e.kind == "door"]
    checks: list[PrincipleCheck] = []
    for bed in beds:
        violators: list[str] = []
        for door in doors:
            wall = room_geom.walls.get(str(door.raw.get("wall_id", "")))
            if wall is None:
                continue
            tx, ty = wall.tangent()
            nx_, ny_ = ty, -tx
            ray = LineString(
                [
                    (door.cx + nx_ * 30.0, door.cy + ny_ * 30.0),
                    (door.cx - nx_ * 30.0, door.cy - ny_ * 30.0),
                ]
            )
            if ray.intersects(bed.footprint()):
                violators.append(door.id)
        if violators:
            checks.append(
                PrincipleCheck(
                    principle="bed_aligned_with_door",
                    target=bed.id,
                    status="violated",
                    score=0.0,
                    evidence=[
                        f"bed {bed.id} is on the direct axis of door(s) {', '.join(violators)}"
                    ],
                )
            )
        else:
            checks.append(
                PrincipleCheck(
                    principle="bed_aligned_with_door",
                    target=bed.id,
                    status="good",
                    score=1.0,
                    evidence=[f"bed {bed.id} is not on a direct door axis"],
                )
            )
    return checks


def _principle_mirror_faces_bed(
    graph: nx.MultiDiGraph,
    room_geom: RoomGeometry,
    zones: list[Zone],
    cfg: GraphConfig,
) -> list[PrincipleCheck]:
    mirrors = [
        e
        for e in room_geom.entities.values()
        if e.kind in ("object", "furniture") and e.label.lower() == "mirror"
    ]
    beds = [
        e
        for e in room_geom.entities.values()
        if e.kind in ("object", "furniture") and e.label.lower() in BED_LABELS
    ]
    checks: list[PrincipleCheck] = []
    for mirror in mirrors:
        fx, fy = mirror.forward_axis_world()
        ray = LineString(
            [
                (mirror.cx, mirror.cy),
                (mirror.cx + fx * 30.0, mirror.cy + fy * 30.0),
            ]
        )
        hit_beds = [b.id for b in beds if ray.intersects(b.footprint())]
        if hit_beds:
            checks.append(
                PrincipleCheck(
                    principle="mirror_faces_bed",
                    target=mirror.id,
                    status="violated",
                    score=0.0,
                    evidence=[
                        f"mirror {mirror.id} reflects toward bed(s) {', '.join(hit_beds)}"
                    ],
                )
            )
    return checks


def _principle_clear_center(
    graph: nx.MultiDiGraph,
    room_geom: RoomGeometry,
    zones: list[Zone],
    cfg: GraphConfig,
) -> list[PrincipleCheck]:
    centroid = room_geom.room_polygon.centroid
    disk = Point(centroid.x, centroid.y).buffer(cfg.clear_center_radius_m)
    disk_inside = disk.intersection(room_geom.room_polygon)
    if disk_inside.area <= 0:
        return []
    blockers: list[tuple[str, float]] = []
    for ent in room_geom.furniture():
        if ent.is_soft():
            continue
        inter = disk_inside.intersection(ent.footprint())
        if inter.area > 0:
            blockers.append((ent.id, inter.area))
    blocked_area = sum(area for _, area in blockers)
    occ = blocked_area / disk_inside.area if disk_inside.area > 0 else 0.0
    evidence = [f"central disk occupancy {occ:.0%}"]
    if blockers:
        top = sorted(blockers, key=lambda x: -x[1])[:5]
        evidence.append(
            "central blockers: "
            + ", ".join(f"{eid}({area:.2f}m^2)" for eid, area in top)
        )
    if occ >= cfg.clear_center_max_occupancy:
        status = "violated"
        score = max(0.0, 1.0 - occ)
    elif occ > cfg.clear_center_max_occupancy * 0.5:
        status = "weak"
        score = max(0.0, 1.0 - occ)
    else:
        status = "good"
        score = 1.0 - occ

    return [
        PrincipleCheck(
            principle="clear_center",
            target="room_0",
            status=status,
            score=float(score),
            evidence=evidence,
        )
    ]


def _principle_sharp_corner(
    graph: nx.MultiDiGraph,
    room_geom: RoomGeometry,
    zones: list[Zone],
    cfg: GraphConfig,
) -> list[PrincipleCheck]:
    primary_seats: list[EntityGeom] = []
    for zone in zones:
        if zone.type != "seating_group":
            continue
        sid = zone.attrs.get("primary_seat_id")
        if sid and sid in room_geom.entities:
            primary_seats.append(room_geom.entities[sid])

    checks: list[PrincipleCheck] = []
    for seat in primary_seats:
        fwd = seat.forward_axis_world()
        offenders: list[tuple[str, float, float]] = []
        for ent in room_geom.furniture():
            if ent.id == seat.id:
                continue
            if ent.is_soft():
                continue
            for cx, cy in ent.corners():
                d = math.hypot(cx - seat.cx, cy - seat.cy)
                if d > cfg.sharp_corner_max_dist_m:
                    continue
                vec = (cx - seat.cx, cy - seat.cy)
                ang = angle_between_deg(fwd, vec)
                if ang <= cfg.sharp_corner_angle_deg:
                    offenders.append((ent.id, d, ang))
                    break
        if offenders:
            evidence = [
                f"{eid} corner {d:.2f}m away at {ang:.1f}deg from forward axis"
                for eid, d, ang in offenders
            ]
            checks.append(
                PrincipleCheck(
                    principle="sharp_corner_points_at_seat",
                    target=seat.id,
                    status="weak",
                    score=max(0.0, 1.0 - 0.2 * len(offenders)),
                    evidence=evidence,
                )
            )
    return checks


def _bed_side_anchors(bed: EntityGeom) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return centers of the bed's left and right side rectangles, in world space."""

    fwd_local = (0.0, 1.0)
    right_local = (fwd_local[1], -fwd_local[0])
    hw, _hd = bed.half_size
    c, s = math.cos(bed.yaw_rad), math.sin(bed.yaw_rad)

    def to_world(local: tuple[float, float]) -> tuple[float, float]:
        return (
            bed.cx + local[0] * c - local[1] * s,
            bed.cy + local[0] * s + local[1] * c,
        )

    left = to_world((-hw, 0.0))
    right = to_world((hw, 0.0))
    return left, right


def _principle_pairing_balance(
    graph: nx.MultiDiGraph,
    room_geom: RoomGeometry,
    zones: list[Zone],
    cfg: GraphConfig,
) -> list[PrincipleCheck]:
    checks: list[PrincipleCheck] = []
    nightstands = [
        e for e in room_geom.furniture() if e.label.lower() in NIGHTSTAND_LABELS
    ]
    for zone in zones:
        if zone.type != "bed_group":
            continue
        bed_id = zone.attrs.get("bed_id")
        if bed_id is None or bed_id not in room_geom.entities:
            continue
        bed = room_geom.entities[bed_id]
        left_anchor, right_anchor = _bed_side_anchors(bed)
        left_match: str | None = None
        right_match: str | None = None
        for ns in nightstands:
            if ns.id not in zone.members:
                continue
            d_left = math.hypot(ns.cx - left_anchor[0], ns.cy - left_anchor[1])
            d_right = math.hypot(ns.cx - right_anchor[0], ns.cy - right_anchor[1])
            if d_left < d_right and d_left <= cfg.bed_nightstand_max_dist_m + 0.5:
                left_match = ns.id
            elif d_right <= cfg.bed_nightstand_max_dist_m + 0.5:
                right_match = ns.id
        if left_match and right_match:
            checks.append(
                PrincipleCheck(
                    principle="pairing_balance",
                    target=bed.id,
                    status="good",
                    score=1.0,
                    evidence=[
                        f"nightstands flank bed: left={left_match}, right={right_match}"
                    ],
                )
            )
        elif left_match or right_match:
            checks.append(
                PrincipleCheck(
                    principle="pairing_balance",
                    target=bed.id,
                    status="weak",
                    score=0.5,
                    evidence=[
                        f"only one side has a nightstand (left={left_match}, right={right_match})"
                    ],
                )
            )
        else:
            checks.append(
                PrincipleCheck(
                    principle="pairing_balance",
                    target=bed.id,
                    status="violated",
                    score=0.0,
                    evidence=["no nightstands on either side of bed"],
                )
            )
    return checks


def _principle_door_alignment(
    graph: nx.MultiDiGraph,
    room_geom: RoomGeometry,
    zones: list[Zone],
    cfg: GraphConfig,
) -> list[PrincipleCheck]:
    doors = list(room_geom.doors())
    checks: list[PrincipleCheck] = []
    for i, a in enumerate(doors):
        for b in doors[i + 1 :]:
            wa = room_geom.walls.get(str(a.raw.get("wall_id", "")))
            wb = room_geom.walls.get(str(b.raw.get("wall_id", "")))
            if wa is None or wb is None:
                continue
            # If the two doors are on roughly parallel walls and the segment
            # connecting them is roughly perpendicular to both walls, they are
            # "aligned" and qi flows straight through.
            yaw_diff = yaw_diff_modpi_deg(wa.yaw_rad(), wb.yaw_rad())
            if yaw_diff > cfg.door_alignment_deg:
                continue
            connector = (b.cx - a.cx, b.cy - a.cy)
            wa_normal = (math.sin(wa.yaw_rad()), -math.cos(wa.yaw_rad()))
            ang = angle_between_deg(connector, wa_normal)
            ang = min(ang, 180.0 - ang)
            if ang > cfg.door_alignment_deg:
                continue
            line = LineString([(a.cx, a.cy), (b.cx, b.cy)])
            blockers = []
            for ent in room_geom.furniture():
                if ent.is_soft() or ent.height < 0.5:
                    continue
                if line.intersects(ent.footprint()):
                    blockers.append(ent.id)
            evidence = [
                f"doors {a.id} and {b.id} are aligned (wall yaw diff {yaw_diff:.1f}deg, normal err {ang:.1f}deg)",
            ]
            if blockers:
                evidence.append(
                    "qi flow partially blocked by " + ", ".join(blockers)
                )
                status = "weak"
                score = 0.5
            else:
                evidence.append("nothing intercepts the cross-door corridor")
                status = "violated"
                score = 0.0
            checks.append(
                PrincipleCheck(
                    principle="door_alignment",
                    target=a.id,
                    status=status,
                    score=score,
                    evidence=evidence,
                    edges=[{"source": a.id, "target": b.id, "type": "aligned_with_door"}],
                )
            )
    return checks


def _principle_light_window_proximity(
    graph: nx.MultiDiGraph,
    room_geom: RoomGeometry,
    zones: list[Zone],
    cfg: GraphConfig,
) -> list[PrincipleCheck]:
    windows = list(room_geom.windows())
    if not windows:
        return []
    checks: list[PrincipleCheck] = []
    for zone in zones:
        if zone.type not in ("seating_group", "bed_group"):
            continue
        anchor_id: str | None = None
        if zone.type == "seating_group":
            anchor_id = zone.attrs.get("primary_seat_id")
        else:
            anchor_id = zone.attrs.get("bed_id")
        if anchor_id is None or anchor_id not in room_geom.entities:
            continue
        anchor = room_geom.entities[anchor_id]
        nearest = min(
            windows,
            key=lambda w: math.hypot(w.cx - anchor.cx, w.cy - anchor.cy),
        )
        d = math.hypot(nearest.cx - anchor.cx, nearest.cy - anchor.cy)
        if d <= cfg.light_window_max_dist_m:
            status = "good"
            score = 1.0
        elif d <= cfg.light_window_max_dist_m * 1.5:
            status = "weak"
            score = 0.5
        else:
            status = "violated"
            score = 0.0
        checks.append(
            PrincipleCheck(
                principle="light_window_proximity",
                target=anchor_id,
                status=status,
                score=score,
                evidence=[
                    f"nearest window {nearest.id} is {d:.2f}m away (threshold {cfg.light_window_max_dist_m:.1f}m)"
                ],
            )
        )
    return checks


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


_RULES = (
    _principle_command_position,
    _principle_solid_backing,
    _principle_bed_aligned_with_door,
    _principle_mirror_faces_bed,
    _principle_clear_center,
    _principle_sharp_corner,
    _principle_pairing_balance,
    _principle_door_alignment,
    _principle_light_window_proximity,
)


def evaluate_principles(
    graph: nx.MultiDiGraph,
    room_geom: RoomGeometry,
    zones: list[Zone],
    cfg: GraphConfig | None = None,
) -> list[PrincipleCheck]:
    cfg = cfg or room_geom.config
    out: list[PrincipleCheck] = []
    for rule in _RULES:
        out.extend(rule(graph, room_geom, zones, cfg))
    return out


def attach_principle_edges(
    graph: nx.MultiDiGraph, checks: Iterable[PrincipleCheck]
) -> None:
    """Add `principle_*` edges from each target node to the room node so that
    downstream tooling can query violations directly off the graph."""

    for check in checks:
        if check.target not in graph:
            continue
        graph.add_edge(
            check.target,
            "room_0",
            type=f"principle_{check.principle}",
            status=check.status,
            score=check.score,
            evidence_count=len(check.evidence),
        )
