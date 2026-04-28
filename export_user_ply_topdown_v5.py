#!/usr/bin/env python3
"""
Raw user .ply -> pseudo-ScanNet + top-down draggable layout exporter (v5).

v5 is a conservative correction pass over the shared high-recall geometry proposal pipeline.
It is meant to address the real-user-scan failure pattern where v4 produced
more labels but shifted boxes to seemingly random places:

  * object centers/footprints are anchored to the source point coordinates;
  * room enforcement no longer snaps objects to the closest boundary point;
  * optional boundary repair nudges boxes inward along a deterministic vector
    toward the room centroid, preserving relative position as much as possible;
  * labels are recomputed with a better priority order: bed/sofa/table/desk
    before cabinet/shelf, so low large furniture is not mislabeled as storage;
  * door/window proxies use all boundary-near points, not just strict wall-mask
    points, and write debug info when nothing is found;
  * pseudo-ScanNet files and scene_layout.json remain compatible with the
    previous ScanNet/top-down implementation.

This is still geometry-only pseudo-recognition.  For reliable semantic labels,
replace this exporter stage with a ScanNet-trained 3D instance segmentation or a
2D/3D multi-view labeling model, then keep this JSON/editor layer unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, MultiPoint, Point, Polygon
from shapely.ops import nearest_points

from export_scannet_topdown import (
    Object2D,
    SceneLayout,
    polygon_bbox,
    render_layout_png,
    save_layout_json,
)

import user_ply_geometry_common as geom


ARCHITECTURE_LABELS = {"door", "window", "door_candidate", "window_candidate"}
GENERIC_OR_WEAK = {
    "small_object_proposal", "furniture_proposal", "unknown_object", "unknown_furniture",
    "large_furniture", "small_furniture", "table_or_desk", "bed_or_sofa",
    "chair_or_small_furniture", "cabinet_or_shelf", "shelf_or_bookcase",
}


def bbox_corners(cx: float, cy: float, width: float, depth: float, theta: float) -> List[List[float]]:
    hw, hd = width / 2.0, depth / 2.0
    local = np.array([[-hw, -hd], [hw, -hd], [hw, hd], [-hw, hd]], dtype=float)
    c, s = math.cos(theta), math.sin(theta)
    rot = np.array([[c, -s], [s, c]], dtype=float)
    pts = local @ rot.T + np.array([cx, cy], dtype=float)
    return [[float(x), float(y)] for x, y in pts]


def object_poly(obj: Object2D) -> Polygon:
    try:
        poly = Polygon(obj.footprint if obj.footprint else bbox_corners(obj.cx, obj.cy, obj.width, obj.depth, obj.theta))
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly
    except Exception:
        return Polygon()


def recompute_footprint(obj: Object2D) -> None:
    obj.footprint = bbox_corners(obj.cx, obj.cy, obj.width, obj.depth, obj.theta)
    obj.bbox3d_center = [float(obj.cx), float(obj.cy), float((obj.z_min + obj.z_max) / 2.0)]
    obj.bbox3d_size = [float(obj.width), float(obj.depth), float(max(obj.height, obj.z_max - obj.z_min, 0.0))]


def wrap_angle(theta: float) -> float:
    while theta <= -math.pi:
        theta += 2.0 * math.pi
    while theta > math.pi:
        theta -= 2.0 * math.pi
    return float(theta)


def snap_to_angle_increment(theta: float, increment_deg: float) -> float:
    inc = math.radians(max(float(increment_deg), 1e-6))
    return wrap_angle(round(float(theta) / inc) * inc)


def fit_existing_footprint_to_angle(obj: Object2D, theta: float, min_size: float = 0.04) -> None:
    """Snap orientation while preserving/enclosing the current footprint.

    Earlier versions changed theta but kept the old OBB extents. For diagonal or
    fragmented scans this could rotate a box into a wall or make it look offset.
    This recomputes width/depth in the snapped coordinate frame so the cardinal
    box encloses the currently exported footprint.
    """
    pts = np.asarray(obj.footprint if obj.footprint else bbox_corners(obj.cx, obj.cy, obj.width, obj.depth, obj.theta), dtype=float)
    if pts.ndim != 2 or len(pts) < 3:
        obj.theta = float(theta)
        recompute_footprint(obj)
        return
    center = pts.mean(axis=0)
    c, ss = math.cos(theta), math.sin(theta)
    R = np.array([[c, -ss], [ss, c]], dtype=float)
    local = (pts - center) @ R
    mn = local.min(axis=0)
    mx = local.max(axis=0)
    local_center = (mn + mx) / 2.0
    world_center = center + local_center @ R.T
    obj.cx = float(world_center[0])
    obj.cy = float(world_center[1])
    obj.width = float(max(mx[0] - mn[0], min_size))
    obj.depth = float(max(mx[1] - mn[1], min_size))
    obj.theta = float(theta)
    recompute_footprint(obj)


def snap_layout_angles_v5(layout: SceneLayout, args: argparse.Namespace) -> Dict[str, object]:
    """Snap exported box orientations to cardinal-style angles by default.

    The resize step is important: it prevents the snapped box from clipping into
    walls simply because a diagonal PCA box was forced to 0/90 degrees.
    """
    debug = {"enabled": bool(args.snap_cardinal_angles), "increment_deg": float(args.snap_angle_increment_deg), "objects": []}
    if not args.snap_cardinal_angles:
        return debug

    for obj in layout.objects:
        is_arch = obj.label in ARCHITECTURE_LABELS
        if is_arch and not args.snap_architecture_angles:
            continue
        old = float(obj.theta)
        old_center = [float(obj.cx), float(obj.cy)]
        old_size = [float(obj.width), float(obj.depth)]
        new = snap_to_angle_increment(old, args.snap_angle_increment_deg)
        if abs(wrap_angle(new - old)) > math.radians(0.1):
            fit_existing_footprint_to_angle(obj, new)
            debug["objects"].append({
                "id": obj.id,
                "label": obj.label,
                "old_theta": old,
                "new_theta": new,
                "old_center": old_center,
                "new_center": [obj.cx, obj.cy],
                "old_size": old_size,
                "new_size": [obj.width, obj.depth],
            })
    debug["changed_count"] = len(debug["objects"])
    return debug


def label_specificity(label: str) -> float:
    return {
        "bed": 4.0, "sofa": 4.0, "table": 3.6, "desk": 3.7, "work_desk": 3.8,
        "kitchen_counter": 3.8, "stove": 3.8, "drawer": 3.4,
        "cabinet_or_shelf": 3.2, "shelf_or_bookcase": 3.2, "chair_or_stool": 2.8,
        "floor_lamp_or_tall_thin_object": 2.2, "small_object": 0.6,
        "small_object_proposal": 0.3, "furniture_proposal": 0.4, "unknown_object": 0.2,
    }.get((label or "").lower(), 1.0)


def object_priority(obj: Object2D) -> float:
    area = float(max(obj.width, 0.0) * max(obj.depth, 0.0))
    point_bonus = min(math.log1p(max(int(getattr(obj, "point_count", 0)), 0)) / 8.0, 1.0)
    height_bonus = min(max(float(obj.height), 0.0), 2.0) * 0.08
    return label_specificity(obj.label) + min(area, 3.0) * 0.25 + point_bonus + height_bonus


def suppress_duplicate_objects_v5(layout: SceneLayout, room_poly: Polygon, args: argparse.Namespace) -> Dict[str, object]:
    """Remove duplicate tiny fragments after label/position cleanup."""
    debug = {"enabled": bool(args.suppress_duplicate_objects), "input_count": len(layout.objects), "discarded": []}
    if not args.suppress_duplicate_objects:
        return debug

    arch = [o for o in layout.objects if o.label in ARCHITECTURE_LABELS]
    furniture = [o for o in layout.objects if o.label not in ARCHITECTURE_LABELS]
    furniture.sort(key=lambda o: object_priority(o), reverse=True)

    kept: List[Object2D] = []
    for obj in furniture:
        area = float(max(obj.width, 0.0) * max(obj.depth, 0.0))
        if area < args.v5_min_final_area and obj.label != "floor_lamp_or_tall_thin_object":
            debug["discarded"].append({"id": obj.id, "label": obj.label, "reason": "below_min_final_area", "area": area})
            continue

        p = object_poly(obj)
        discard_reason = None
        discard_against = None
        for prev in kept:
            q = object_poly(prev)
            if p.is_empty or q.is_empty:
                continue
            inter = p.intersection(q).area
            if inter <= 0:
                dist = math.hypot(obj.cx - prev.cx, obj.cy - prev.cy)
                same_weak_family = (
                    obj.label == prev.label
                    or obj.label in GENERIC_OR_WEAK
                    or prev.label in GENERIC_OR_WEAK
                    or obj.label == "floor_lamp_or_tall_thin_object" == prev.label
                )
                if same_weak_family and min(area, prev.width * prev.depth) <= args.v5_fragment_area and dist <= args.v5_duplicate_center_distance:
                    discard_reason = "nearby_small_fragment_duplicate"
                    discard_against = prev.id
                    break
                continue

            iou = inter / max(p.union(q).area, 1e-9)
            containment = inter / max(min(p.area, q.area), 1e-9)
            if iou >= args.v5_duplicate_iou or containment >= args.v5_duplicate_containment:
                discard_reason = "overlap_or_containment_duplicate"
                discard_against = prev.id
                break

        if discard_reason is not None:
            debug["discarded"].append({"id": obj.id, "label": obj.label, "reason": discard_reason, "against": discard_against, "area": area})
        else:
            kept.append(obj)

    kept.sort(key=lambda o: (round(o.cy, 3), round(o.cx, 3), o.label.lower()))
    layout.objects = kept + arch
    for i, obj in enumerate(layout.objects, start=1):
        obj.id = i
        if obj.label in ARCHITECTURE_LABELS:
            obj.object_id = i
    debug["output_count"] = len(layout.objects)
    debug["discarded_count"] = len(debug["discarded"])
    return debug


def boundary_distance(room_poly: Polygon, x: float, y: float) -> float:
    if room_poly.is_empty:
        return float("inf")
    return float(room_poly.exterior.distance(Point(x, y)))


def point_inside_or_near(room_poly: Polygon, x: float, y: float, tol: float) -> bool:
    if room_poly.is_empty:
        return True
    return bool(room_poly.buffer(tol).contains(Point(x, y)))


def area_ratio_outside(obj: Object2D, room_poly: Polygon, tol: float) -> float:
    poly = object_poly(obj)
    if poly.is_empty or poly.area <= 1e-9 or room_poly.is_empty:
        return 0.0
    outside = poly.difference(room_poly.buffer(tol)).area
    return float(outside / max(poly.area, 1e-9))


def classify_geometry_v5(obj: Object2D, room_poly: Polygon) -> Tuple[str, float, str]:
    """Geometry-only label guess with better priority ordering.

    This is still a heuristic, but v5.1 biases away from overusing
    cabinet/shelf and floor-lamp labels. Large footprints and tabletop/counter
    footprints are classified before any storage-like label, even when clutter on
    top raises z_max.
    """
    w = float(max(obj.width, 1e-6))
    d = float(max(obj.depth, 1e-6))
    h = float(max(obj.height, obj.z_max - obj.z_min, 0.0))
    zmin = float(obj.z_min)
    zmax = float(obj.z_max)
    area = w * d
    long_side = max(w, d)
    short_side = min(w, d)
    aspect = long_side / max(short_side, 1e-6)
    near_wall = boundary_distance(room_poly, obj.cx, obj.cy) <= 0.60
    old = (obj.label or "").lower()
    raw = (obj.raw_label or "").lower()

    if old in ARCHITECTURE_LABELS:
        return obj.label, 0.80, "preserved_architecture"

    # Bed/sofa-like footprints must win before cabinet/shelf.  Real scans often
    # include blankets, cushions, books, or other clutter on top, so zmax can be
    # too high for a naive "low object" rule.
    if long_side >= 1.70 and short_side >= 0.72 and area >= 1.25 and zmin <= 0.55:
        if long_side >= 1.85 and short_side >= 0.82 and area >= 1.55:
            return "bed", 0.66, "bed_sized_footprint_even_with_top_clutter"
        return "sofa", 0.55, "sofa_sized_footprint_even_with_top_clutter"

    # Partial couch/bed scans can come out as long but thin fragments.
    if long_side >= 1.25 and short_side >= 0.38 and area >= 0.62 and zmin <= 0.65 and zmax <= 1.65:
        if near_wall and aspect >= 2.4:
            return "sofa", 0.46, "long_low_wall_adjacent_sofa_fragment"
        return "sofa", 0.42, "partial_large_low_furniture_fragment"

    # Counter/desk/table-height surfaces.  This comes before storage so tables
    # with monitors/items on top do not become cabinet_or_shelf.
    table_or_counter_candidate = (
        0.28 <= area <= 3.80
        and long_side >= 0.65
        and short_side >= 0.22
        and zmin <= 1.20
        and zmax <= 1.85
    )
    if table_or_counter_candidate:
        if near_wall and long_side >= 1.05:
            if long_side >= 1.45 and short_side <= 0.95:
                # Use counter for long, wall-attached surfaces; otherwise desk.
                if zmax >= 0.80 or "counter" in raw:
                    return "kitchen_counter", 0.54, "long_wall_adjacent_counter_or_countertop_surface"
                return "desk", 0.50, "long_wall_adjacent_desk_surface"
            return "desk", 0.46, "wall_adjacent_table_height_surface"
        if long_side >= 0.75:
            return "table", 0.50, "free_standing_table_surface"

    # Chair/stool-like objects.
    if 0.06 <= area <= 0.75 and 0.20 <= zmax <= 1.35 and long_side <= 1.10:
        return "chair_or_stool", 0.36, "small_sitting_height_object"

    # Floor lamp / tall thin objects are now deliberately strict.  Many previous
    # false positives were fragments from sofas/beds/desks.
    if (
        area <= 0.09
        and long_side <= 0.42
        and h >= 0.85
        and zmin <= 0.35
        and zmax >= 1.15
    ):
        return "floor_lamp_or_tall_thin_object", 0.42, "strict_small_footprint_floor_to_tall_object"

    # Storage only if it is actually tall and not a plausible table/bed footprint.
    if (zmax >= 1.45 or h >= 1.20) and area >= 0.16:
        if near_wall or aspect >= 1.25:
            return "cabinet_or_shelf", 0.36, "tall_wall_or_rectangular_storage_like"
        return "shelf_or_bookcase", 0.33, "tall_free_standing_storage_like"

    # Medium object fallback, still more useful than *_proposal.
    if area >= 0.35:
        if near_wall:
            return "desk", 0.28, "generic_medium_wall_adjacent"
        return "table", 0.27, "generic_medium_free_standing"

    return "small_object", 0.12, "too_small_or_sparse_for_specific_label"


def relabel_objects_v5(layout: SceneLayout, room_poly: Polygon) -> Dict[str, object]:
    changes = []
    for obj in layout.objects:
        old = obj.label
        new, conf, reason = classify_geometry_v5(obj, room_poly)
        obj.label = new
        obj.raw_label = f"{old}__v5_refined_to__{new}__{reason}__conf_{conf:.2f}"
        changes.append({"id": obj.id, "old": old, "new": new, "confidence": conf, "reason": reason})
    return {"changes": changes}


def is_fragment_candidate_v5(obj: Object2D, args: argparse.Namespace) -> bool:
    area = float(max(obj.width, 0.0) * max(obj.depth, 0.0))
    label = (obj.label or "").lower()
    if label in ARCHITECTURE_LABELS:
        return False
    if area <= args.fragment_merge_max_member_area:
        return True
    if label in {"small_object", "small_object_proposal", "furniture_proposal", "floor_lamp_or_tall_thin_object", "unknown_object"}:
        return True
    return False


def polygons_close_or_overlap(a: Polygon, b: Polygon, max_dist: float) -> bool:
    if a.is_empty or b.is_empty:
        return False
    if a.intersects(b):
        return True
    return float(a.distance(b)) <= max_dist


def merge_fragment_objects_v5(layout: SceneLayout, room_poly: Polygon, args: argparse.Namespace) -> Dict[str, object]:
    """Merge nearby tiny/generic fragments into one larger anchored furniture box.

    This fixes a common raw-PLY failure mode where one couch/bed/desk becomes a
    row of small_object or floor_lamp_or_tall_thin_object pieces.  The merge is
    deliberately conservative: it only merges nearby weak fragments, and the
    merged box is cardinal/axis-aligned so it stays editable and wall-friendly.
    """
    debug = {"enabled": bool(args.merge_fragments), "merged_components": [], "input_count": len(layout.objects)}
    if not args.merge_fragments:
        return debug

    objs = [o for o in layout.objects if o.label not in ARCHITECTURE_LABELS]
    arch = [o for o in layout.objects if o.label in ARCHITECTURE_LABELS]
    n = len(objs)
    if n == 0:
        return debug

    polys = [object_poly(o) for o in objs]
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        if not is_fragment_candidate_v5(objs[i], args):
            continue
        for j in range(i + 1, n):
            if not is_fragment_candidate_v5(objs[j], args):
                continue
            if polygons_close_or_overlap(polys[i], polys[j], args.fragment_merge_distance):
                union(i, j)

    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)

    used = set()
    new_objs: List[Object2D] = []
    next_temp_id = 1
    for root, members in groups.items():
        if len(members) < 2:
            continue
        member_objs = [objs[i] for i in members]
        total_area = sum(max(o.width, 0.0) * max(o.depth, 0.0) for o in member_objs)
        if total_area < args.fragment_merge_min_total_area:
            continue

        # Build an axis/cardinal-aligned union box from the current footprints.
        pts_list = []
        for o in member_objs:
            fp = np.asarray(o.footprint if o.footprint else bbox_corners(o.cx, o.cy, o.width, o.depth, o.theta), dtype=float)
            if fp.ndim == 2 and len(fp) >= 3:
                pts_list.append(fp)
        if not pts_list:
            continue
        pts2 = np.vstack(pts_list)
        mn = pts2.min(axis=0)
        mx = pts2.max(axis=0)
        w = float(max(mx[0] - mn[0], 0.05))
        d = float(max(mx[1] - mn[1], 0.05))
        area = w * d
        if area > args.fragment_merge_max_merged_area:
            continue
        if not room_poly.is_empty and area / max(float(room_poly.area), 1e-6) > args.fragment_merge_max_room_ratio:
            continue

        merged = Object2D(
            id=next_temp_id,
            object_id=min(int(o.object_id) for o in member_objs),
            label="merged_furniture",
            raw_label="v5_merged_from_fragments:" + ",".join(f"{o.id}:{o.label}" for o in member_objs),
            cx=float((mn[0] + mx[0]) / 2.0),
            cy=float((mn[1] + mx[1]) / 2.0),
            width=w,
            depth=d,
            theta=0.0,
            z_min=float(min(o.z_min for o in member_objs)),
            z_max=float(max(o.z_max for o in member_objs)),
            height=float(max(o.z_max for o in member_objs) - min(o.z_min for o in member_objs)),
            point_count=int(sum(int(getattr(o, "point_count", 0)) for o in member_objs)),
            footprint=[],
            bbox3d_center=[0.0, 0.0, 0.0],
            bbox3d_size=[0.0, 0.0, 0.0],
        )
        recompute_footprint(merged)
        new_label, conf, reason = classify_geometry_v5(merged, room_poly)
        # Do not replace several fragments with another vague tiny label.
        if new_label in {"small_object", "floor_lamp_or_tall_thin_object"} and area < 0.60:
            continue
        merged.label = new_label
        merged.raw_label += f"__classified_as_{new_label}__{reason}__conf_{conf:.2f}"
        new_objs.append(merged)
        used.update(members)
        debug["merged_components"].append({
            "member_ids": [objs[i].id for i in members],
            "member_labels": [objs[i].label for i in members],
            "new_label": new_label,
            "width": w,
            "depth": d,
            "area": area,
            "confidence": conf,
            "reason": reason,
        })
        next_temp_id += 1

    kept_unmerged = [o for i, o in enumerate(objs) if i not in used]
    layout.objects = kept_unmerged + new_objs + arch
    for i, obj in enumerate(layout.objects, start=1):
        obj.id = i
    debug["merged_count"] = len(new_objs)
    debug["removed_fragment_count"] = len(used)
    debug["output_count"] = len(layout.objects)
    return debug


def nudge_object_inside(obj: Object2D, room_poly: Polygon, tol: float, max_outside_ratio: float) -> Tuple[bool, str]:
    """Move an object inward without random nearest-boundary snapping.

    The shift direction is from the current center toward the room centroid.  This
    preserves relative layout better than snapping to an arbitrary nearest point.
    """
    if room_poly.is_empty:
        return True, "no_room_polygon"
    if area_ratio_outside(obj, room_poly, tol) <= max_outside_ratio and point_inside_or_near(room_poly, obj.cx, obj.cy, tol):
        return True, "already_inside_or_near"

    start = np.array([obj.cx, obj.cy], dtype=float)
    target = np.array([room_poly.representative_point().x, room_poly.representative_point().y], dtype=float)
    vec = target - start
    n = float(np.linalg.norm(vec))
    if n < 1e-8:
        return False, "cannot_compute_inward_vector"
    vec /= n

    best = (float("inf"), obj.cx, obj.cy)
    max_step = max(room_poly.bounds[2] - room_poly.bounds[0], room_poly.bounds[3] - room_poly.bounds[1], 1.0)
    for step in np.linspace(0.0, max_step, 60):
        obj.cx = float(start[0] + vec[0] * step)
        obj.cy = float(start[1] + vec[1] * step)
        recompute_footprint(obj)
        ratio = area_ratio_outside(obj, room_poly, tol)
        center_ok = point_inside_or_near(room_poly, obj.cx, obj.cy, tol)
        score = ratio + (0.25 if not center_ok else 0.0) + 0.02 * step
        if score < best[0]:
            best = (score, obj.cx, obj.cy)
        if ratio <= max_outside_ratio and center_ok:
            return True, "nudged_toward_room_centroid"

    obj.cx = float(best[1])
    obj.cy = float(best[2])
    recompute_footprint(obj)
    if area_ratio_outside(obj, room_poly, tol) <= max_outside_ratio * 1.5:
        return True, "best_effort_nudge"
    return False, "still_outside_after_nudge"


def enforce_positions_v5(layout: SceneLayout, room_poly: Polygon, args: argparse.Namespace) -> Dict[str, object]:
    kept: List[Object2D] = []
    room_area = max(float(room_poly.area), 1e-6)
    debug = {
        "policy": args.position_policy,
        "input_count": len(layout.objects),
        "rejected_room_sized": 0,
        "rejected_outside": 0,
        "nudged": 0,
        "preserved_outside": 0,
        "objects": [],
    }

    for obj in layout.objects:
        if obj.label in ARCHITECTURE_LABELS:
            kept.append(obj)
            continue
        area = float(max(obj.width, 0.0) * max(obj.depth, 0.0))
        if area / room_area > args.v5_max_box_area_ratio or area > args.max_box_area:
            debug["rejected_room_sized"] += 1
            debug["objects"].append({"id": obj.id, "label": obj.label, "action": "reject_room_sized", "area": area})
            continue

        ratio_before = area_ratio_outside(obj, room_poly, args.v5_room_tolerance)
        center_before = [obj.cx, obj.cy]
        action = "kept"
        ok = True
        if ratio_before > args.v5_max_outside_ratio or not point_inside_or_near(room_poly, obj.cx, obj.cy, args.v5_room_tolerance):
            if args.position_policy == "preserve":
                debug["preserved_outside"] += 1
                action = "preserved_even_if_outside"
            elif args.position_policy == "reject":
                ok = False
                action = "reject_outside"
                debug["rejected_outside"] += 1
            elif args.position_policy == "nudge":
                ok, action = nudge_object_inside(obj, room_poly, args.v5_room_tolerance, args.v5_max_outside_ratio)
                if ok and action != "already_inside_or_near":
                    debug["nudged"] += 1
                elif not ok:
                    debug["rejected_outside"] += 1
            else:
                raise ValueError(f"Unknown position policy: {args.position_policy}")

        if ok:
            kept.append(obj)
        debug["objects"].append({
            "id": obj.id,
            "label": obj.label,
            "action": action,
            "center_before": center_before,
            "center_after": [obj.cx, obj.cy],
            "outside_ratio_before": ratio_before,
            "outside_ratio_after": area_ratio_outside(obj, room_poly, args.v5_room_tolerance),
        })

    # Spatial ID ordering is easier to compare against the original scan than
    # size-based sorting.
    kept.sort(key=lambda o: (round(o.cy, 3), round(o.cx, 3), o.label.lower()))
    for i, obj in enumerate(kept, start=1):
        obj.id = i
        # Preserve object_id for furniture so pseudo-ScanNet segGroups still map
        # back to their original generated segment/object IDs.
    layout.objects = kept
    debug["output_count"] = len(kept)
    return debug


def line_theta(p0: np.ndarray, p1: np.ndarray) -> float:
    return float(math.atan2(float(p1[1] - p0[1]), float(p1[0] - p0[0])))


def make_arch_obj(obj_id: int, label: str, center: np.ndarray, length: float, thickness: float, theta: float, zmin: float, zmax: float, reason: str) -> Object2D:
    return Object2D(
        id=obj_id,
        object_id=obj_id,
        label=label,
        raw_label=f"v5_architecture_proxy__{reason}",
        cx=float(center[0]),
        cy=float(center[1]),
        width=float(length),
        depth=float(thickness),
        theta=float(theta),
        z_min=float(zmin),
        z_max=float(zmax),
        height=float(max(0.0, zmax - zmin)),
        point_count=0,
        footprint=bbox_corners(float(center[0]), float(center[1]), float(length), float(thickness), float(theta)),
        bbox3d_center=[float(center[0]), float(center[1]), float((zmin + zmax) / 2.0)],
        bbox3d_size=[float(length), float(thickness), float(max(0.0, zmax - zmin))],
    )


def detect_architecture_v5(aligned: np.ndarray, room_poly: Polygon, args: argparse.Namespace, start_id: int) -> Tuple[List[Object2D], Dict[str, object]]:
    if not args.detect_architecture or room_poly.is_empty or len(room_poly.exterior.coords) < 4:
        return [], {"enabled": False}

    coords = np.asarray(room_poly.exterior.coords[:-1], dtype=float)
    proxies: List[Object2D] = []
    edge_debug: List[Dict[str, object]] = []
    obj_id = start_id
    room_center = np.array([room_poly.representative_point().x, room_poly.representative_point().y], dtype=float)

    for edge_i in range(len(coords)):
        p0 = coords[edge_i]
        p1 = coords[(edge_i + 1) % len(coords)]
        vec = p1 - p0
        edge_len = float(np.linalg.norm(vec))
        if edge_len < args.arch_min_wall_len:
            continue
        unit = vec / max(edge_len, 1e-9)
        rel = aligned[:, :2] - p0
        t = rel @ unit
        proj = p0 + np.outer(t, unit)
        dist = np.linalg.norm(aligned[:, :2] - proj, axis=1)
        near = (t >= 0) & (t <= edge_len) & (dist <= args.arch_wall_distance)
        pts = aligned[near]
        ts = t[near]
        if len(pts) < args.arch_min_edge_points:
            edge_debug.append({"edge": edge_i, "length": edge_len, "near_points": int(len(pts)), "skipped": "too_few_points"})
            continue

        nb = max(4, int(math.ceil(edge_len / args.arch_bin_width)))
        bins = np.clip((ts / edge_len * nb).astype(int), 0, nb - 1)
        z = pts[:, 2]

        low_counts = np.bincount(bins[(z >= 0.05) & (z <= 0.70)], minlength=nb).astype(float)
        mid_counts = np.bincount(bins[(z >= args.arch_window_min_z) & (z <= args.arch_window_max_z)], minlength=nb).astype(float)
        high_counts = np.bincount(bins[(z >= 1.35) & (z <= args.arch_door_max_z)], minlength=nb).astype(float)
        full_counts = low_counts + mid_counts + high_counts

        nonzero_full = full_counts[full_counts > 0]
        nonzero_mid = mid_counts[mid_counts > 0]
        med_full = float(np.median(nonzero_full)) if len(nonzero_full) else 0.0
        med_mid = float(np.median(nonzero_mid)) if len(nonzero_mid) else 0.0
        if med_full <= 0:
            edge_debug.append({"edge": edge_i, "length": edge_len, "near_points": int(len(pts)), "skipped": "no_wall_evidence"})
            continue

        def runs(mask: np.ndarray) -> List[Tuple[int, int]]:
            out: List[Tuple[int, int]] = []
            i = 0
            while i < len(mask):
                if not mask[i]:
                    i += 1
                    continue
                j = i
                while j + 1 < len(mask) and mask[j + 1]:
                    j += 1
                out.append((i, j))
                i = j + 1
            return out

        theta = line_theta(p0, p1)
        midpoint = (p0 + p1) / 2.0
        inward = room_center - midpoint
        inward /= max(float(np.linalg.norm(inward)), 1e-9)

        added = 0
        # Door: full vertical gap, but only when surrounded by evidence.
        gap_thresh = max(1.0, med_full * args.arch_gap_fraction)
        door_gap = full_counts <= gap_thresh
        for a, b in runs(door_gap):
            width = (b - a + 1) * edge_len / nb
            if not (args.arch_door_min_width <= width <= args.arch_door_max_width):
                continue
            left = full_counts[max(0, a - 2):a]
            right = full_counts[b + 1:min(nb, b + 3)]
            side_support = (left.sum() + right.sum())
            if side_support < max(2.0, med_full * 0.75):
                continue
            mid_t = ((a + b + 1) / 2.0) / nb * edge_len
            center = p0 + unit * mid_t + inward * (args.arch_proxy_thickness * 0.85)
            proxies.append(make_arch_obj(obj_id, "door", center, width, args.arch_proxy_thickness, theta, 0.0, args.arch_door_max_z, "vertical_wall_gap"))
            obj_id += 1
            added += 1
            if added >= args.arch_max_openings_per_edge:
                break

        # Window: mid-height gap with low or high wall support in same bins.
        if med_mid > 0:
            window_gap = mid_counts <= max(1.0, med_mid * args.arch_gap_fraction)
            window_added = 0
            for a, b in runs(window_gap):
                width = (b - a + 1) * edge_len / nb
                if not (args.arch_window_min_width <= width <= args.arch_window_max_width):
                    continue
                support = low_counts[a:b + 1].sum() + high_counts[a:b + 1].sum()
                if support < max(2.0, med_full * 0.40):
                    continue
                mid_t = ((a + b + 1) / 2.0) / nb * edge_len
                center = p0 + unit * mid_t + inward * (args.arch_proxy_thickness * 0.85)
                proxies.append(make_arch_obj(obj_id, "window", center, width, args.arch_proxy_thickness, theta, args.arch_window_min_z, args.arch_window_max_z, "mid_height_wall_gap"))
                obj_id += 1
                window_added += 1
                if window_added >= args.arch_max_openings_per_edge:
                    break
            added += window_added

        edge_debug.append({
            "edge": edge_i,
            "length": edge_len,
            "near_points": int(len(pts)),
            "bins": int(nb),
            "median_full": med_full,
            "median_mid": med_mid,
            "added": int(added),
        })

    # Deduplicate same-label overlapping proxies.
    deduped: List[Object2D] = []
    for obj in proxies:
        p = object_poly(obj)
        duplicate = False
        for prev in deduped:
            if prev.label != obj.label:
                continue
            q = object_poly(prev)
            inter = p.intersection(q).area if not p.is_empty and not q.is_empty else 0.0
            if inter / max(min(p.area, q.area), 1e-9) > 0.45:
                duplicate = True
                break
        if not duplicate:
            deduped.append(obj)

    def add_fallback_candidate(label: str, edge_i: int, frac: float, width: float, zmin: float, zmax: float, reason: str) -> bool:
        nonlocal obj_id
        p0 = coords[edge_i]
        p1 = coords[(edge_i + 1) % len(coords)]
        edge_len = float(np.linalg.norm(p1 - p0))
        if edge_len < max(width * 1.05, args.arch_min_wall_len):
            return False
        unit = (p1 - p0) / max(edge_len, 1e-9)
        center = p0 + unit * (edge_len * min(max(frac, 0.05), 0.95))
        inward = room_center - center
        inward /= max(float(np.linalg.norm(inward)), 1e-9)
        center = center + inward * (args.arch_proxy_thickness * 0.85)
        candidate = make_arch_obj(
            obj_id,
            label,
            center,
            min(width, edge_len * 0.80),
            args.arch_proxy_thickness,
            line_theta(p0, p1),
            zmin,
            zmax,
            reason,
        )
        cand_poly = object_poly(candidate)
        for prev in deduped:
            prev_poly = object_poly(prev)
            if cand_poly.is_empty or prev_poly.is_empty:
                continue
            inter = cand_poly.intersection(prev_poly).area
            if inter / max(min(cand_poly.area, prev_poly.area), 1e-9) > 0.25:
                return False
        deduped.append(candidate)
        obj_id += 1
        return True

    # Fallback architectural proxies: raw PLY scans often do not contain enough
    # clean wall-gap evidence for reliable semantic detection of doors/windows.
    # These are marked as candidates so downstream UI can expose them as editable.
    if args.arch_add_candidate_if_none and len(coords) >= 2:
        lengths = [float(np.linalg.norm(coords[(i + 1) % len(coords)] - coords[i])) for i in range(len(coords))]
        order = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)

        fallback_door_label = "door_candidate" if args.arch_fallback_as_candidate else "door"
        fallback_window_label = "window_candidate" if args.arch_fallback_as_candidate else "window"

        door_count = sum(1 for o in deduped if o.label in {"door", "door_candidate"})
        for edge_i in order:
            if door_count >= args.arch_min_door_candidates:
                break
            if add_fallback_candidate(
                fallback_door_label,
                edge_i,
                0.50,
                min(args.arch_fallback_door_width, max(args.arch_door_min_width, lengths[edge_i] * 0.35)),
                0.0,
                args.arch_door_max_z,
                "fallback_long_wall_door_from_floor_to_above_2m",
            ):
                door_count += 1

        window_count = sum(1 for o in deduped if o.label in {"window", "window_candidate"})
        frac_cycle = [0.32, 0.68, 0.50]
        frac_i = 0
        for edge_i in order:
            if window_count >= args.arch_min_window_candidates:
                break
            width = min(args.arch_fallback_window_width, max(args.arch_window_min_width, lengths[edge_i] * 0.35))
            if add_fallback_candidate(
                fallback_window_label,
                edge_i,
                frac_cycle[frac_i % len(frac_cycle)],
                width,
                args.arch_window_min_z,
                args.arch_window_max_z,
                "fallback_long_wall_window_from_hip_height_to_below_ceiling",
            ):
                window_count += 1
                frac_i += 1

    deduped = deduped[: args.arch_max_total]
    for i, obj in enumerate(deduped, start=start_id):
        obj.id = i
        obj.object_id = i
    return deduped, {"enabled": True, "raw_proxy_count": len(proxies), "output_proxy_count": len(deduped), "edges": edge_debug}


def sync_pseudo_with_layout(pseudo: Dict[str, object], layout: SceneLayout) -> None:
    by_id = {int(obj.object_id): obj.label for obj in layout.objects if obj.label not in ARCHITECTURE_LABELS}
    synced = []
    for g in pseudo.get("seg_groups", []):
        oid = int(g.get("objectId", g.get("id", -1)))
        if oid in by_id:
            g["label"] = by_id[oid]
            synced.append(g)
    pseudo["seg_groups"] = synced


def build_layout_v5(args: argparse.Namespace):
    # Start from source-point-anchored geometry proposals, so centers stay anchored to the scan coordinates.
    # High recall is now opt-in because the previous default exported too many tiny duplicate fragments.
    if args.v5_high_recall:
        args.filter_outside_room = False
        args.wall_filter = args.wall_filter or "high_only"
        args.max_objects = max(args.max_objects, 160)
        args.min_object_points = min(args.min_object_points, 6)
        args.min_box_area = min(args.min_box_area, 0.0015)
        args.min_object_height = min(args.min_object_height, 0.02)

    layout, pseudo, aligned, rgb, debug = geom.build_layout(args)
    room_poly = Polygon(layout.room_polygon).buffer(0)
    if room_poly.is_empty or not room_poly.is_valid:
        room_poly = MultiPoint(np.asarray(layout.room_polygon, dtype=float)).convex_hull

    label_debug = relabel_objects_v5(layout, room_poly)
    snap_debug = snap_layout_angles_v5(layout, args)
    merge_debug = merge_fragment_objects_v5(layout, room_poly, args)

    # Re-run label + snap after merging because the merged footprint can now look
    # like a bed/sofa/table/desk even if all original pieces were weak labels.
    label_debug_after_merge = relabel_objects_v5(layout, room_poly)
    snap_debug_after_merge = snap_layout_angles_v5(layout, args)

    position_debug = enforce_positions_v5(layout, room_poly, args)
    duplicate_debug = suppress_duplicate_objects_v5(layout, room_poly, args)

    arch_objects, arch_debug = detect_architecture_v5(aligned, room_poly, args, start_id=len(layout.objects) + 1)
    if args.snap_cardinal_angles and args.snap_architecture_angles and arch_objects:
        tmp_layout = SceneLayout(scene_id=layout.scene_id, units=layout.units, room_polygon=layout.room_polygon, room_bbox=layout.room_bbox, objects=arch_objects, metadata={})
        snap_layout_angles_v5(tmp_layout, args)
    layout.objects.extend(arch_objects)

    # Keep spatial order for easier visual comparison.
    furniture = [o for o in layout.objects if o.label not in ARCHITECTURE_LABELS]
    architecture = [o for o in layout.objects if o.label in ARCHITECTURE_LABELS]
    furniture.sort(key=lambda o: (round(o.cy, 3), round(o.cx, 3), o.label.lower()))
    architecture.sort(key=lambda o: (o.label, round(o.cy, 3), round(o.cx, 3)))
    layout.objects = furniture + architecture
    for i, obj in enumerate(layout.objects, start=1):
        obj.id = i
        if obj.label in ARCHITECTURE_LABELS:
            obj.object_id = i

    layout.room_bbox = polygon_bbox(room_poly)
    layout.metadata.update({
        "source": "user_ply_geometry_pseudo_scannet_v5",
        "v5_changes": [
            "anchored object positions from source-point geometry proposals",
            "no nearest-boundary snapping",
            "optional centroid-based inward nudging/rejection/preservation",
            "label priority changed so bed/sofa/table/desk beat cabinet/shelf",
            "default cardinal angle snapping for cleaner editable boxes",
            "post-refinement duplicate/fragment suppression",
            "door/window proxies from boundary-near point gaps plus candidate fallbacks",
        ],
        "v5_position_policy": args.position_policy,
        "v5_label_refinement": label_debug,
        "v5_angle_snapping": snap_debug,
        "v5_fragment_merging": merge_debug,
        "v5_label_refinement_after_merge": label_debug_after_merge,
        "v5_angle_snapping_after_merge": snap_debug_after_merge,
        "v5_position_enforcement": position_debug,
        "v5_duplicate_suppression": duplicate_debug,
        "v5_architecture_detection": arch_debug,
        "v5_warning": "Labels and doors/windows are geometry-only guesses. Use editor correction or ML segmentation for reliable semantics.",
    })
    debug.update({
        "v5_label_refinement": label_debug,
        "v5_fragment_merging": merge_debug,
        "v5_label_refinement_after_merge": label_debug_after_merge,
        "v5_position_enforcement": position_debug,
        "v5_architecture_detection": arch_debug,
        "v5_object_labels": [o.label for o in layout.objects],
    })
    sync_pseudo_with_layout(pseudo, layout)
    return layout, pseudo, aligned, rgb, debug


def write_debug_v5(out_dir: str | Path, layout: SceneLayout, aligned: np.ndarray, pseudo: Dict[str, object], debug: Dict[str, object], seed: int) -> None:
    geom.write_debug(out_dir, layout, aligned, pseudo, debug, seed)
    out_dir = Path(out_dir)

    room = np.asarray(layout.room_polygon, dtype=float)
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.scatter(aligned[:, 0], aligned[:, 1], s=0.25, alpha=0.10, label="aligned raw points")
    if len(room) >= 3:
        closed = np.vstack([room, room[0]])
        ax.plot(closed[:, 0], closed[:, 1], linewidth=2.0, label="room polygon")
    for obj in layout.objects:
        corners = np.asarray(obj.footprint, dtype=float)
        if len(corners) < 3:
            continue
        closed = np.vstack([corners, corners[0]])
        lw = 2.2 if obj.label in ARCHITECTURE_LABELS else 1.1
        ax.plot(closed[:, 0], closed[:, 1], linewidth=lw)
        ax.scatter([obj.cx], [obj.cy], s=10)
        ax.text(obj.cx, obj.cy, f"{obj.id}:{obj.label}", fontsize=7)
    ax.axis("equal")
    ax.legend(loc="best")
    ax.set_title("v5 final: raw-point anchored boxes + refined labels")
    fig.tight_layout()
    fig.savefig(out_dir / "debug_v5_final_points_overlay.png", dpi=180)
    plt.close(fig)

    with (out_dir / "debug_v5_summary.json").open("w", encoding="utf-8") as f:
        json.dump(debug, f, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Raw user .ply -> pseudo-ScanNet + top-down draggable layout exporter v5")

    # Shared geometry-compatible arguments.
    p.add_argument("--ply", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--scene_id", default=None)

    p.add_argument("--up_axis", default="z", choices=["auto", "x", "y", "z"])
    p.add_argument("--assume_aligned", action="store_true")
    p.add_argument("--plane_distance", type=float, default=0.04)
    p.add_argument("--ransac_iterations", type=int, default=900)

    p.add_argument("--floor_height_threshold", type=float, default=0.06)
    p.add_argument("--room_source", default="auto", choices=["auto", "floor", "lower_slice"])
    p.add_argument("--room_slice_max_z", type=float, default=0.35)
    p.add_argument("--room_resolution_m", type=float, default=0.035)
    p.add_argument("--room_close_iters", type=int, default=0)
    p.add_argument("--room_dilate_iters", type=int, default=0)
    p.add_argument("--room_simplify_m", type=float, default=0.003)
    p.add_argument("--min_room_area", type=float, default=0.50)

    p.add_argument("--wall_filter", default="high_only", choices=["none", "high_only", "hard"])
    p.add_argument("--wall_filter_min_z", type=float, default=1.45)
    p.add_argument("--wall_detect_max_z", type=float, default=2.40)
    p.add_argument("--ceiling_min_z", type=float, default=0.0)

    p.add_argument("--object_min_z", type=float, default=0.06)
    p.add_argument("--object_max_z", type=float, default=2.15)
    p.add_argument("--low_band_max_z", type=float, default=0.95)
    p.add_argument("--table_band_min_z", type=float, default=0.35)
    p.add_argument("--table_band_max_z", type=float, default=1.25)
    p.add_argument("--counter_band_min_z", type=float, default=0.62)
    p.add_argument("--counter_band_max_z", type=float, default=1.40)
    p.add_argument("--tall_band_min_z", type=float, default=0.90)
    p.add_argument("--tall_band_max_z", type=float, default=2.15)

    p.add_argument("--occupancy_resolution", type=float, default=0.035)
    p.add_argument("--object_close_iters", type=int, default=0)
    p.add_argument("--object_dilate_iters", type=int, default=0)
    p.add_argument("--min_component_cells", type=int, default=2)
    p.add_argument("--min_object_points", type=int, default=18)
    p.add_argument("--min_object_height", type=float, default=0.02)
    p.add_argument("--max_object_height", type=float, default=2.40)
    p.add_argument("--min_box_area", type=float, default=0.012)
    p.add_argument("--max_box_area", type=float, default=8.0)
    p.add_argument("--max_box_area_ratio", type=float, default=0.35)
    p.add_argument("--max_box_long_side_ratio", type=float, default=0.90)
    p.add_argument("--box_padding", type=float, default=0.08)
    p.add_argument("--box_trim_percentile", type=float, default=0.5)
    p.add_argument("--filter_outside_room", action="store_true")
    p.add_argument("--outside_room_tolerance", type=float, default=0.30)

    p.add_argument("--nms_iou", type=float, default=0.35)
    p.add_argument("--nms_containment", type=float, default=0.55)
    p.add_argument("--max_objects", type=int, default=60)

    # v5 behavior.
    p.add_argument("--v5_high_recall", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--position_policy", default="nudge", choices=["preserve", "nudge", "reject"], help="How to handle boxes outside the room. v5 uses centroid-nudge, not nearest-boundary snapping.")
    p.add_argument("--v5_room_tolerance", type=float, default=0.08)
    p.add_argument("--v5_max_outside_ratio", type=float, default=0.25)
    p.add_argument("--v5_max_box_area_ratio", type=float, default=0.24)

    # Angle/duplicate cleanup.
    p.add_argument("--snap_cardinal_angles", action=argparse.BooleanOptionalAction, default=True, help="Snap furniture boxes to the nearest 0/90/180/270 degree angle by default.")
    p.add_argument("--snap_angle_increment_deg", type=float, default=90.0)
    p.add_argument("--snap_architecture_angles", action=argparse.BooleanOptionalAction, default=False, help="Also snap door/window proxy orientations; default keeps them on room-boundary edges.")
    p.add_argument("--suppress_duplicate_objects", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--v5_duplicate_iou", type=float, default=0.15)
    p.add_argument("--v5_duplicate_containment", type=float, default=0.35)
    p.add_argument("--v5_duplicate_center_distance", type=float, default=0.50)
    p.add_argument("--v5_fragment_area", type=float, default=0.30)
    p.add_argument("--v5_min_final_area", type=float, default=0.018)

    # Fragment cleanup/merge. These defaults prefer fewer duplicated tiny boxes
    # and larger usable furniture proposals for the editor.
    p.add_argument("--merge_fragments", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fragment_merge_distance", type=float, default=0.38)
    p.add_argument("--fragment_merge_max_member_area", type=float, default=0.36)
    p.add_argument("--fragment_merge_min_total_area", type=float, default=0.38)
    p.add_argument("--fragment_merge_max_merged_area", type=float, default=4.50)
    p.add_argument("--fragment_merge_max_room_ratio", type=float, default=0.22)

    # Architecture proxy detection.
    p.add_argument("--detect_architecture", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--arch_add_candidate_if_none", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--arch_fallback_as_candidate", action=argparse.BooleanOptionalAction, default=False,
                   help="If false, fallback architectural proxies are labeled as door/window instead of door_candidate/window_candidate.")
    p.add_argument("--arch_wall_distance", type=float, default=0.35)
    p.add_argument("--arch_bin_width", type=float, default=0.10)
    p.add_argument("--arch_gap_fraction", type=float, default=0.55)
    p.add_argument("--arch_min_wall_len", type=float, default=0.80)
    p.add_argument("--arch_min_edge_points", type=int, default=5)
    p.add_argument("--arch_proxy_thickness", type=float, default=0.12)
    p.add_argument("--arch_door_min_width", type=float, default=0.45)
    p.add_argument("--arch_door_max_width", type=float, default=1.60)
    p.add_argument("--arch_door_max_z", type=float, default=2.10)
    p.add_argument("--arch_window_min_z", type=float, default=0.75)
    p.add_argument("--arch_window_max_z", type=float, default=1.85)
    p.add_argument("--arch_window_min_width", type=float, default=0.40)
    p.add_argument("--arch_window_max_width", type=float, default=2.60)
    p.add_argument("--arch_max_openings_per_edge", type=int, default=2)
    p.add_argument("--arch_min_door_candidates", type=int, default=2)
    p.add_argument("--arch_min_window_candidates", type=int, default=2)
    p.add_argument("--arch_fallback_door_width", type=float, default=0.85)
    p.add_argument("--arch_fallback_window_width", type=float, default=1.20)
    p.add_argument("--arch_max_total", type=int, default=12)

    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--no_png", action="store_true")
    p.add_argument("--no_debug", action="store_true")
    p.add_argument("--no_copy_original", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layout, pseudo, aligned, rgb, debug = build_layout_v5(args)
    save_layout_json(layout, out_dir / "scene_layout.json")
    if not args.no_png:
        render_layout_png(layout, out_dir / "scene_layout.png")
    scene_id = args.scene_id or Path(args.ply).stem.replace(" ", "_")
    pseudo_dir = geom.write_pseudo_scannet_files(out_dir, scene_id, args.ply, aligned, rgb, pseudo, copy_original=not args.no_copy_original)
    if not args.no_debug:
        write_debug_v5(out_dir, layout, aligned, pseudo, debug, args.seed)

    print(f"Saved layout JSON to: {out_dir / 'scene_layout.json'}")
    if not args.no_png:
        print(f"Saved layout preview to: {out_dir / 'scene_layout.png'}")
    if not args.no_debug:
        print(f"Saved debug summary to: {out_dir / 'debug_detection_summary.json'}")
        print(f"Saved v5 debug summary to: {out_dir / 'debug_v5_summary.json'}")
        print(f"Saved v5 final overlay to: {out_dir / 'debug_v5_final_points_overlay.png'}")
    print(f"Saved pseudo-ScanNet scene folder to: {pseudo_dir}")
    print(f"Objects exported: {len(layout.objects)}")


if __name__ == "__main__":
    main()
