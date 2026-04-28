#!/usr/bin/env python3
"""
Raw user .ply -> pseudo-ScanNet + top-down draggable layout exporter (v5).

v5 is a conservative correction pass over the high-recall v3 proposal pipeline.
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

import export_user_ply_topdown_v3 as v3


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

    The most important change from v4 is that cabinet/shelf is only chosen when
    the object is actually tall.  Large low rectangles and table-height surfaces
    are evaluated first.
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
    near_wall = boundary_distance(room_poly, obj.cx, obj.cy) <= 0.55
    old = (obj.label or "").lower()
    raw = (obj.raw_label or "").lower()

    # Preserve actual architectural labels, but not furniture guesses.
    if old in ARCHITECTURE_LABELS:
        return obj.label, 0.75, "preserved_architecture"

    # Very thin/tall: lamp, plant, coat rack, stand.
    if area <= 0.12 and long_side <= 0.60 and zmax >= 1.05:
        return "floor_lamp_or_tall_thin_object", 0.50, "small_footprint_tall_object"

    # Large low furniture wins before storage.  Sparse phone scans often capture
    # only the surface shell, so height/zmax can be imperfect; footprint matters.
    if long_side >= 1.65 and short_side >= 0.70 and area >= 1.20 and zmax <= 1.25:
        if long_side >= 1.80 and short_side >= 0.85 and area >= 1.55:
            return "bed", 0.56, "large_low_bed_sized_footprint"
        return "sofa", 0.48, "large_low_sofa_sized_footprint"

    # Sofa-like but slightly smaller/partial scan.
    if long_side >= 1.20 and short_side >= 0.45 and area >= 0.80 and zmax <= 1.20:
        return "sofa", 0.40, "partial_large_low_sofa_like_footprint"

    # Counter/desk/table-height surfaces.  These are checked before cabinet.
    table_height = (0.45 <= zmax <= 1.30) or (0.45 <= zmin <= 1.10) or (0.25 <= h <= 1.05 and zmax <= 1.35)
    if table_height and 0.18 <= area <= 3.20:
        if near_wall and long_side >= 1.10 and short_side <= 0.95:
            if long_side >= 1.40 and short_side <= 0.80 and ("counter" in raw or zmax >= 0.75):
                return "kitchen_counter", 0.48, "long_wall_adjacent_counter_height_surface"
            if long_side >= 0.95:
                return "desk", 0.45, "wall_adjacent_table_height_surface"
        if long_side >= 0.85 and short_side >= 0.35:
            return "table", 0.44, "free_or_central_table_height_surface"

    # Work desk can appear as a long partial tabletop near a wall, even with low height span.
    if near_wall and long_side >= 1.05 and 0.25 <= area <= 2.50 and zmax <= 1.35:
        return "desk", 0.38, "long_wall_adjacent_surface"

    # Chair/stool-like objects.
    if 0.06 <= area <= 0.75 and 0.25 <= zmax <= 1.35 and long_side <= 1.10:
        return "chair_or_stool", 0.34, "small_sitting_height_object"

    # Storage only if actually tall. This prevents bed/table -> cabinet_or_shelf.
    if (zmax >= 1.35 or h >= 1.10) and area >= 0.12:
        if near_wall or aspect >= 1.25:
            return "cabinet_or_shelf", 0.38, "tall_wall_or_rectangular_storage_like"
        return "shelf_or_bookcase", 0.34, "tall_free_standing_storage_like"

    # Medium object fallback, still more useful than *_proposal.
    if area >= 0.35:
        if near_wall:
            return "desk", 0.25, "generic_medium_wall_adjacent"
        return "table", 0.24, "generic_medium_free_standing"

    return "small_object", 0.16, "too_small_or_sparse_for_specific_label"


def relabel_objects_v5(layout: SceneLayout, room_poly: Polygon) -> Dict[str, object]:
    changes = []
    for obj in layout.objects:
        old = obj.label
        new, conf, reason = classify_geometry_v5(obj, room_poly)
        obj.label = new
        obj.raw_label = f"{old}__v5_refined_to__{new}__{reason}__conf_{conf:.2f}"
        changes.append({"id": obj.id, "old": old, "new": new, "confidence": conf, "reason": reason})
    return {"changes": changes}


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
        obj.object_id = i
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

    # Optional fallback: create a door_candidate at the longest boundary edge.
    # This is clearly marked as a candidate so users know it needs correction.
    if args.arch_add_candidate_if_none and not any(o.label == "door" for o in deduped) and len(coords) >= 2:
        lengths = [float(np.linalg.norm(coords[(i + 1) % len(coords)] - coords[i])) for i in range(len(coords))]
        edge_i = int(np.argmax(lengths))
        p0 = coords[edge_i]
        p1 = coords[(edge_i + 1) % len(coords)]
        edge_len = lengths[edge_i]
        if edge_len >= args.arch_door_min_width:
            unit = (p1 - p0) / max(edge_len, 1e-9)
            center = (p0 + p1) / 2.0
            inward = room_center - center
            inward /= max(float(np.linalg.norm(inward)), 1e-9)
            center = center + inward * (args.arch_proxy_thickness * 0.85)
            deduped.append(make_arch_obj(obj_id, "door_candidate", center, min(0.85, edge_len * 0.40), args.arch_proxy_thickness, line_theta(p0, p1), 0.0, args.arch_door_max_z, "fallback_longest_wall_candidate_needs_user_confirmation"))

    deduped = deduped[: args.arch_max_total]
    for i, obj in enumerate(deduped, start=start_id):
        obj.id = i
        obj.object_id = i
    return deduped, {"enabled": True, "raw_proxy_count": len(proxies), "output_proxy_count": len(deduped), "edges": edge_debug}


def sync_pseudo_with_layout(pseudo: Dict[str, object], layout: SceneLayout) -> None:
    by_id = {int(obj.object_id): obj.label for obj in layout.objects if obj.label not in ARCHITECTURE_LABELS}
    for g in pseudo.get("seg_groups", []):
        oid = int(g.get("objectId", g.get("id", -1)))
        if oid in by_id:
            g["label"] = by_id[oid]


def build_layout_v5(args: argparse.Namespace):
    # Start from v3 proposals, not v4, so centers stay anchored to source points.
    if args.v5_high_recall:
        args.filter_outside_room = False
        args.wall_filter = args.wall_filter or "high_only"
        args.max_objects = max(args.max_objects, 160)
        args.min_object_points = min(args.min_object_points, 6)
        args.min_box_area = min(args.min_box_area, 0.0015)
        args.min_object_height = min(args.min_object_height, 0.02)

    layout, pseudo, aligned, rgb, debug = v3.build_layout(args)
    room_poly = Polygon(layout.room_polygon).buffer(0)
    if room_poly.is_empty or not room_poly.is_valid:
        room_poly = MultiPoint(np.asarray(layout.room_polygon, dtype=float)).convex_hull

    label_debug = relabel_objects_v5(layout, room_poly)
    position_debug = enforce_positions_v5(layout, room_poly, args)

    arch_objects, arch_debug = detect_architecture_v5(aligned, room_poly, args, start_id=len(layout.objects) + 1)
    layout.objects.extend(arch_objects)

    # Keep spatial order for easier visual comparison.
    furniture = [o for o in layout.objects if o.label not in ARCHITECTURE_LABELS]
    architecture = [o for o in layout.objects if o.label in ARCHITECTURE_LABELS]
    furniture.sort(key=lambda o: (round(o.cy, 3), round(o.cx, 3), o.label.lower()))
    architecture.sort(key=lambda o: (o.label, round(o.cy, 3), round(o.cx, 3)))
    layout.objects = furniture + architecture
    for i, obj in enumerate(layout.objects, start=1):
        obj.id = i
        obj.object_id = i

    layout.room_bbox = polygon_bbox(room_poly)
    layout.metadata.update({
        "source": "user_ply_geometry_pseudo_scannet_v5",
        "v5_changes": [
            "anchored object positions from v3 source-point proposals",
            "no nearest-boundary snapping",
            "optional centroid-based inward nudging/rejection/preservation",
            "label priority changed so bed/sofa/table/desk beat cabinet/shelf",
            "door/window proxies from boundary-near point gaps",
        ],
        "v5_position_policy": args.position_policy,
        "v5_label_refinement": label_debug,
        "v5_position_enforcement": position_debug,
        "v5_architecture_detection": arch_debug,
        "v5_warning": "Labels and doors/windows are geometry-only guesses. Use editor correction or ML segmentation for reliable semantics.",
    })
    debug.update({
        "v5_label_refinement": label_debug,
        "v5_position_enforcement": position_debug,
        "v5_architecture_detection": arch_debug,
        "v5_object_labels": [o.label for o in layout.objects],
    })
    sync_pseudo_with_layout(pseudo, layout)
    return layout, pseudo, aligned, rgb, debug


def write_debug_v5(out_dir: str | Path, layout: SceneLayout, aligned: np.ndarray, pseudo: Dict[str, object], debug: Dict[str, object], seed: int) -> None:
    v3.write_debug(out_dir, layout, aligned, pseudo, debug, seed)
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

    # v3-compatible arguments.
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
    p.add_argument("--min_component_cells", type=int, default=1)
    p.add_argument("--min_object_points", type=int, default=6)
    p.add_argument("--min_object_height", type=float, default=0.02)
    p.add_argument("--max_object_height", type=float, default=2.40)
    p.add_argument("--min_box_area", type=float, default=0.0015)
    p.add_argument("--max_box_area", type=float, default=8.0)
    p.add_argument("--max_box_area_ratio", type=float, default=0.35)
    p.add_argument("--max_box_long_side_ratio", type=float, default=0.90)
    p.add_argument("--box_padding", type=float, default=0.08)
    p.add_argument("--box_trim_percentile", type=float, default=0.5)
    p.add_argument("--filter_outside_room", action="store_true")
    p.add_argument("--outside_room_tolerance", type=float, default=0.30)

    p.add_argument("--nms_iou", type=float, default=0.50)
    p.add_argument("--nms_containment", type=float, default=0.80)
    p.add_argument("--max_objects", type=int, default=160)

    # v5 behavior.
    p.add_argument("--v5_high_recall", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--position_policy", default="nudge", choices=["preserve", "nudge", "reject"], help="How to handle boxes outside the room. v5 uses centroid-nudge, not nearest-boundary snapping.")
    p.add_argument("--v5_room_tolerance", type=float, default=0.08)
    p.add_argument("--v5_max_outside_ratio", type=float, default=0.25)
    p.add_argument("--v5_max_box_area_ratio", type=float, default=0.24)

    # Architecture proxy detection.
    p.add_argument("--detect_architecture", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--arch_add_candidate_if_none", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--arch_wall_distance", type=float, default=0.22)
    p.add_argument("--arch_bin_width", type=float, default=0.14)
    p.add_argument("--arch_gap_fraction", type=float, default=0.28)
    p.add_argument("--arch_min_wall_len", type=float, default=0.80)
    p.add_argument("--arch_min_edge_points", type=int, default=12)
    p.add_argument("--arch_proxy_thickness", type=float, default=0.12)
    p.add_argument("--arch_door_min_width", type=float, default=0.45)
    p.add_argument("--arch_door_max_width", type=float, default=1.60)
    p.add_argument("--arch_door_max_z", type=float, default=2.10)
    p.add_argument("--arch_window_min_z", type=float, default=0.75)
    p.add_argument("--arch_window_max_z", type=float, default=1.85)
    p.add_argument("--arch_window_min_width", type=float, default=0.40)
    p.add_argument("--arch_window_max_width", type=float, default=2.60)
    p.add_argument("--arch_max_openings_per_edge", type=int, default=2)
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
    pseudo_dir = v3.write_pseudo_scannet_files(out_dir, scene_id, args.ply, aligned, rgb, pseudo, copy_original=not args.no_copy_original)
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
