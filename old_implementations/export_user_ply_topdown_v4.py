#!/usr/bin/env python3
"""
Raw user .ply -> pseudo-ScanNet + top-down draggable room layout exporter (v4).

v4 is an extension layer over v3.  It keeps the same output contract as the
ScanNet exporter and v3 exporter, but adds post-processing that is more useful
for real phone/LiDAR room scans:

  * upgrades generic *_proposal labels into more useful geometry-based labels;
  * merges nearby small fragments into larger furniture boxes when plausible;
  * rejects or clips room-sized boxes and snaps boxes back inside the room;
  * adds approximate door/window architectural proxies from wall/boundary gaps;
  * writes the same scene_layout.json and pseudo-ScanNet folder as before.

Important: this is still geometry-only pseudo-recognition.  It creates better
object proposals, not true furniture recognition.  For reliable labels, replace
this proposal stage with a trained 3D semantic/instance segmentation model.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString, MultiPoint, Point, Polygon
from shapely.ops import nearest_points, unary_union

from export_scannet_topdown import (
    Object2D,
    SceneLayout,
    oriented_bbox_2d,
    polygon_bbox,
    render_layout_png,
    save_layout_json,
)

# Reuse v3's actual parsing, alignment, pseudo-ScanNet writing, and debug output.
import export_user_ply_topdown_v3 as v3


ARCHITECTURE_LABELS = {"door", "window"}
GENERIC_LABELS = {
    "small_object_proposal",
    "furniture_proposal",
    "unknown_object",
    "unknown_furniture",
}


def _poly_from_object(obj: Object2D) -> Polygon:
    try:
        p = Polygon(obj.footprint)
        if not p.is_valid:
            p = p.buffer(0)
        return p
    except Exception:
        return Polygon()


def _bbox_corners(cx: float, cy: float, width: float, depth: float, theta: float) -> List[List[float]]:
    hw, hd = width / 2.0, depth / 2.0
    local = np.array([[-hw, -hd], [hw, -hd], [hw, hd], [-hw, hd]], dtype=float)
    c, s = math.cos(theta), math.sin(theta)
    rot = np.array([[c, -s], [s, c]], dtype=float)
    pts = local @ rot.T + np.array([cx, cy], dtype=float)
    return [[float(x), float(y)] for x, y in pts]


def _obj_area(obj: Object2D) -> float:
    return float(max(obj.width, 0.0) * max(obj.depth, 0.0))


def _room_boundary_distance(room_poly: Polygon, x: float, y: float) -> float:
    if room_poly.is_empty:
        return float("inf")
    return float(room_poly.exterior.distance(Point(x, y)))


def semantic_label_from_geometry(obj: Object2D, room_poly: Polygon) -> Tuple[str, float, str]:
    """Return (label, confidence, reason) from dimensions/height/wall proximity.

    This deliberately upgrades labels before falling back to *_proposal.  It is
    still a guess, but it gives the editor more useful labels for downstream
    feng-shui heuristics.
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
    near_wall = _room_boundary_distance(room_poly, obj.cx, obj.cy) <= 0.55
    raw = (obj.raw_label or "").lower()
    old = (obj.label or "").lower()

    # Preserve already useful labels, unless they are proposal/fallback labels.
    useful = {
        "bed", "sofa", "table", "desk", "work_desk", "kitchen_counter",
        "cabinet_or_shelf", "shelf_or_bookcase", "floor_lamp_or_tall_thin_object",
        "chair_or_stool", "door", "window",
    }
    if old in useful:
        return obj.label, 0.60, "preserved_existing_specific_label"

    # Very small footprint but high vertical object: lamp/plant/stand candidate.
    if area <= 0.16 and h >= 0.75:
        return "floor_lamp_or_tall_thin_object", 0.48, "small_footprint_tall_height"

    # Large low rectangles: bed/sofa.  Height is often underestimated in sparse PLYs,
    # so dimensions are more important than height here.
    if long_side >= 1.65 and short_side >= 0.65 and area >= 1.25 and zmax <= 1.25:
        if long_side >= 1.85 and short_side >= 0.90 and near_wall:
            return "bed", 0.52, "large_low_rectangle_near_wall"
        return "sofa", 0.48, "large_low_rectangle"

    # Long narrow objects along walls are usually desks, counters, or cabinets.
    if near_wall and long_side >= 1.10 and short_side <= 0.95 and area >= 0.35:
        if 0.65 <= zmax <= 1.35 or "counter" in raw:
            if long_side >= 1.40 and short_side <= 0.75:
                return "kitchen_counter", 0.46, "long_counter_height_near_wall"
            return "work_desk", 0.44, "counter_or_desk_height_near_wall"
        if zmax > 1.25:
            return "cabinet_or_shelf", 0.42, "tall_long_object_near_wall"
        return "desk", 0.40, "long_object_near_wall"

    # Table/desk surfaces.  Many phone scans capture only tabletop fragments, so
    # accept modest heights/areas.
    if 0.18 <= area <= 2.50 and 0.35 <= zmax <= 1.35:
        if near_wall and long_side >= 0.85:
            return "desk", 0.39, "table_height_near_wall"
        return "table", 0.38, "table_height_free_standing"

    # Chairs/stools are moderate height and small-medium footprint.
    if 0.08 <= area <= 0.75 and 0.25 <= zmax <= 1.35:
        return "chair_or_stool", 0.33, "small_medium_object_sitting_height"

    # Tall storage.
    if h >= 1.10 or zmax >= 1.45:
        if near_wall or aspect >= 1.25:
            return "cabinet_or_shelf", 0.36, "tall_storage_like"
        return "shelf_or_bookcase", 0.34, "tall_free_standing_storage_like"

    if area >= 0.35:
        if near_wall:
            return "desk", 0.25, "generic_medium_near_wall"
        return "table", 0.24, "generic_medium_free_standing"

    return "unknown_furniture", 0.16, "insufficient_geometry_for_specific_label"


def refine_labels(layout: SceneLayout, room_poly: Polygon, preserve_proposal_labels: bool = False) -> None:
    for obj in layout.objects:
        if obj.label in ARCHITECTURE_LABELS:
            continue
        new_label, confidence, reason = semantic_label_from_geometry(obj, room_poly)
        if preserve_proposal_labels and obj.label not in GENERIC_LABELS and not obj.label.endswith("_proposal"):
            continue
        old_label = obj.label
        obj.label = new_label
        obj.raw_label = f"{old_label}__v4_refined_to__{new_label}__{reason}__conf_{confidence:.2f}"


def oriented_box_from_objects(objects: List[Object2D], new_id: int, room_poly: Polygon) -> Object2D:
    pts: List[List[float]] = []
    zmin = float("inf")
    zmax = float("-inf")
    point_count = 0
    for obj in objects:
        pts.extend(obj.footprint)
        zmin = min(zmin, obj.z_min)
        zmax = max(zmax, obj.z_max)
        point_count += int(max(obj.point_count, 0))
    arr = np.asarray(pts, dtype=float)
    center, width, depth, theta, corners = oriented_bbox_2d(arr)
    height = max(0.0, zmax - zmin)
    merged = Object2D(
        id=new_id,
        object_id=new_id,
        label="unknown_furniture",
        raw_label="v4_merged_fragment_candidate",
        cx=float(center[0]),
        cy=float(center[1]),
        width=float(width),
        depth=float(depth),
        theta=float(theta),
        z_min=float(zmin if np.isfinite(zmin) else 0.0),
        z_max=float(zmax if np.isfinite(zmax) else height),
        height=float(height),
        point_count=int(point_count),
        footprint=[[float(x), float(y)] for x, y in corners.tolist()],
        bbox3d_center=[float(center[0]), float(center[1]), float((zmin + zmax) / 2.0 if np.isfinite(zmin + zmax) else 0.0)],
        bbox3d_size=[float(width), float(depth), float(height)],
    )
    new_label, conf, reason = semantic_label_from_geometry(merged, room_poly)
    merged.label = new_label
    merged.raw_label = f"v4_merged_{len(objects)}_fragments__{reason}__conf_{conf:.2f}"
    return merged


def merge_fragment_objects(
    objects: List[Object2D],
    room_poly: Polygon,
    merge_distance: float,
    min_merged_area: float,
    max_merged_area_ratio: float,
) -> Tuple[List[Object2D], Dict[str, object]]:
    """Merge nearby small/generic proposals into larger furniture candidates.

    Phone/LiDAR scans often capture a bed/sofa/desk as multiple disconnected
    pieces.  This pass only merges generic or small fragments; already specific
    labels are preserved unless they lie inside a larger merged proposal.
    """
    if merge_distance <= 0 or not objects:
        return objects, {"enabled": False}

    room_area = max(float(room_poly.area), 1e-6)
    candidates: List[Tuple[int, Polygon]] = []
    preserve: List[Object2D] = []
    for i, obj in enumerate(objects):
        area = _obj_area(obj)
        generic = obj.label in GENERIC_LABELS or obj.label.endswith("_proposal") or obj.raw_label.startswith("small_object")
        smallish = area <= 1.25
        if obj.label in ARCHITECTURE_LABELS:
            preserve.append(obj)
            continue
        if generic or smallish:
            poly = _poly_from_object(obj)
            if not poly.is_empty:
                candidates.append((i, poly.buffer(merge_distance / 2.0)))
            else:
                preserve.append(obj)
        else:
            preserve.append(obj)

    if not candidates:
        return objects, {"enabled": True, "candidate_count": 0, "merged_groups": 0}

    # Build connected groups using buffered polygon intersections.
    parent = list(range(len(candidates)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a in range(len(candidates)):
        for b in range(a + 1, len(candidates)):
            if candidates[a][1].intersects(candidates[b][1]):
                union(a, b)

    groups: Dict[int, List[int]] = {}
    for local_idx, (obj_idx, _) in enumerate(candidates):
        groups.setdefault(find(local_idx), []).append(obj_idx)

    used: set[int] = set()
    merged_objects: List[Object2D] = []
    next_id = 1
    debug_groups = []
    for group in groups.values():
        group_objs = [objects[i] for i in group]
        if len(group_objs) < 2:
            continue
        merged = oriented_box_from_objects(group_objs, new_id=next_id, room_poly=room_poly)
        area = _obj_area(merged)
        if area < min_merged_area or area / room_area > max_merged_area_ratio:
            continue
        merged_objects.append(merged)
        used.update(group)
        debug_groups.append({
            "fragment_count": len(group_objs),
            "label": merged.label,
            "area": area,
            "source_ids": [o.id for o in group_objs],
        })
        next_id += 1

    result = []
    result.extend(preserve)
    for i, obj in enumerate(objects):
        if i in used:
            continue
        if obj in preserve:
            continue
        result.append(obj)
    result.extend(merged_objects)

    # Re-number after merging.
    for new_i, obj in enumerate(result, start=1):
        obj.id = new_i
        obj.object_id = new_i
    return result, {
        "enabled": True,
        "candidate_count": len(candidates),
        "merged_groups": len(merged_objects),
        "groups": debug_groups,
    }


def enforce_room_boundary(
    objects: List[Object2D],
    room_poly: Polygon,
    max_area_ratio: float,
    outside_tolerance: float,
    shrink_to_fit: bool,
    snap_centers: bool,
) -> Tuple[List[Object2D], Dict[str, object]]:
    kept: List[Object2D] = []
    room_area = max(float(room_poly.area), 1e-6)
    debug = {"rejected_room_sized": 0, "rejected_outside": 0, "snapped_centers": 0, "shrunk_boxes": 0}

    for obj in objects:
        if obj.label in ARCHITECTURE_LABELS:
            kept.append(obj)
            continue
        area = _obj_area(obj)
        if area / room_area > max_area_ratio:
            debug["rejected_room_sized"] += 1
            continue

        center = Point(obj.cx, obj.cy)
        if snap_centers and not room_poly.buffer(outside_tolerance).contains(center):
            nearest = nearest_points(room_poly, center)[0]
            obj.cx = float(nearest.x)
            obj.cy = float(nearest.y)
            obj.footprint = _bbox_corners(obj.cx, obj.cy, obj.width, obj.depth, obj.theta)
            debug["snapped_centers"] += 1

        poly = _poly_from_object(obj)
        if poly.is_empty:
            continue
        outside_area = poly.difference(room_poly.buffer(outside_tolerance)).area
        if outside_area > max(0.20 * poly.area, 0.05):
            if shrink_to_fit:
                # Shrink around the center until most of the footprint lies inside.
                original_w, original_d = obj.width, obj.depth
                for _ in range(18):
                    obj.width *= 0.94
                    obj.depth *= 0.94
                    obj.footprint = _bbox_corners(obj.cx, obj.cy, obj.width, obj.depth, obj.theta)
                    poly = _poly_from_object(obj)
                    outside_area = poly.difference(room_poly.buffer(outside_tolerance)).area
                    if outside_area <= max(0.20 * poly.area, 0.05):
                        debug["shrunk_boxes"] += 1
                        break
                # If we had to shrink too much, the proposal was probably bad.
                if obj.width < 0.35 * original_w or obj.depth < 0.35 * original_d:
                    debug["rejected_outside"] += 1
                    continue
            else:
                debug["rejected_outside"] += 1
                continue
        kept.append(obj)

    for new_i, obj in enumerate(kept, start=1):
        obj.id = new_i
        obj.object_id = new_i
    return kept, debug


def _line_orientation_theta(p0: np.ndarray, p1: np.ndarray) -> float:
    return float(math.atan2(float(p1[1] - p0[1]), float(p1[0] - p0[0])))


def _make_arch_object(
    obj_id: int,
    label: str,
    center: np.ndarray,
    length: float,
    thickness: float,
    theta: float,
    zmin: float,
    zmax: float,
    reason: str,
) -> Object2D:
    return Object2D(
        id=obj_id,
        object_id=obj_id,
        label=label,
        raw_label=f"v4_architecture_proxy__{reason}",
        cx=float(center[0]),
        cy=float(center[1]),
        width=float(length),
        depth=float(thickness),
        theta=float(theta),
        z_min=float(zmin),
        z_max=float(zmax),
        height=float(max(0.0, zmax - zmin)),
        point_count=0,
        footprint=_bbox_corners(float(center[0]), float(center[1]), float(length), float(thickness), float(theta)),
        bbox3d_center=[float(center[0]), float(center[1]), float((zmin + zmax) / 2.0)],
        bbox3d_size=[float(length), float(thickness), float(max(0.0, zmax - zmin))],
    )


def detect_architecture_proxies(
    aligned: np.ndarray,
    wall_mask: np.ndarray,
    room_poly: Polygon,
    args: argparse.Namespace,
    start_id: int,
) -> Tuple[List[Object2D], Dict[str, object]]:
    """Approximate doors/windows from gaps in wall evidence along room edges.

    This is intentionally conservative.  It does not know true semantics, but it
    gives the feng-shui engine architectural anchors that are better than having
    no door/window information at all.
    """
    if not args.detect_architecture or room_poly.is_empty or len(room_poly.exterior.coords) < 4:
        return [], {"enabled": False}

    coords = np.asarray(room_poly.exterior.coords[:-1], dtype=float)
    wall_pts = aligned[wall_mask]
    if len(wall_pts) < 30:
        return [], {"enabled": True, "reason": "too_few_wall_points", "wall_points": int(len(wall_pts))}

    proxies: List[Object2D] = []
    edge_debug: List[dict] = []
    obj_id = start_id

    for edge_i in range(len(coords)):
        p0 = coords[edge_i]
        p1 = coords[(edge_i + 1) % len(coords)]
        vec = p1 - p0
        length = float(np.linalg.norm(vec))
        if length < args.arch_min_wall_len:
            continue
        unit = vec / max(length, 1e-6)
        rel_xy = wall_pts[:, :2] - p0
        t = rel_xy @ unit
        # perpendicular distance to edge line
        proj = p0 + np.outer(t, unit)
        dist = np.linalg.norm(wall_pts[:, :2] - proj, axis=1)
        near = (t >= 0) & (t <= length) & (dist <= args.arch_wall_distance)
        pts = wall_pts[near]
        ts = t[near]
        if len(pts) < args.arch_min_edge_points:
            continue

        bin_w = args.arch_bin_width
        nb = max(3, int(math.ceil(length / bin_w)))
        bins = np.clip((ts / length * nb).astype(int), 0, nb - 1)
        z = pts[:, 2]
        full_counts = np.bincount(bins[(z >= 0.10) & (z <= args.arch_door_max_z)], minlength=nb)
        mid_counts = np.bincount(bins[(z >= args.arch_window_min_z) & (z <= args.arch_window_max_z)], minlength=nb)
        low_counts = np.bincount(bins[(z >= 0.10) & (z <= 0.75)], minlength=nb)
        high_counts = np.bincount(bins[(z >= args.arch_window_max_z) & (z <= args.arch_door_max_z)], minlength=nb)

        median_full = float(np.median(full_counts[full_counts > 0])) if np.any(full_counts > 0) else 0.0
        median_mid = float(np.median(mid_counts[mid_counts > 0])) if np.any(mid_counts > 0) else 0.0
        if median_full <= 0:
            continue

        # Door-like: near-empty vertical wall column from low to door height,
        # surrounded by bins with wall evidence.
        door_empty = full_counts <= max(1.0, median_full * args.arch_gap_fraction)
        window_empty = mid_counts <= max(1.0, median_mid * args.arch_gap_fraction) if median_mid > 0 else np.zeros(nb, dtype=bool)

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

        theta = _line_orientation_theta(p0, p1)
        # Offset slightly inward toward the room centroid so the proxy is visible.
        centroid = np.asarray([room_poly.centroid.x, room_poly.centroid.y])
        normal = centroid - ((p0 + p1) / 2.0)
        normal = normal / max(np.linalg.norm(normal), 1e-6)

        added_on_edge = 0
        for a, b in runs(door_empty):
            gap_len = (b - a + 1) * length / nb
            if not (args.arch_door_min_width <= gap_len <= args.arch_door_max_width):
                continue
            # Require some wall evidence on either side to avoid making every end
            # of a partial scan into a door.
            left = max(0, a - 2)
            right = min(nb, b + 3)
            side_counts = np.concatenate([full_counts[left:a], full_counts[b + 1:right]])
            if len(side_counts) and np.max(side_counts) < max(2.0, median_full * 0.50):
                continue
            mid_t = ((a + b + 1) / 2.0) / nb * length
            center = p0 + unit * mid_t + normal * (args.arch_proxy_thickness * 0.75)
            proxies.append(_make_arch_object(obj_id, "door", center, gap_len, args.arch_proxy_thickness, theta, 0.0, args.arch_door_max_z, "low_to_high_wall_gap"))
            obj_id += 1
            added_on_edge += 1
            if added_on_edge >= args.arch_max_openings_per_edge:
                break

        # Window-like: mid-height gap but low or high wall evidence remains.
        added_windows = 0
        for a, b in runs(window_empty):
            gap_len = (b - a + 1) * length / nb
            if not (args.arch_window_min_width <= gap_len <= args.arch_window_max_width):
                continue
            support = (low_counts[a:b + 1].sum() + high_counts[a:b + 1].sum())
            if support < max(2.0, median_full * 0.30):
                continue
            mid_t = ((a + b + 1) / 2.0) / nb * length
            center = p0 + unit * mid_t + normal * (args.arch_proxy_thickness * 0.75)
            proxies.append(_make_arch_object(obj_id, "window", center, gap_len, args.arch_proxy_thickness, theta, args.arch_window_min_z, args.arch_window_max_z, "mid_height_wall_gap"))
            obj_id += 1
            added_windows += 1
            if added_windows >= args.arch_max_openings_per_edge:
                break

        edge_debug.append({
            "edge": int(edge_i),
            "length": length,
            "near_wall_points": int(len(pts)),
            "median_full_count": median_full,
            "median_mid_count": median_mid,
            "proxies_added_on_edge": added_on_edge + added_windows,
        })

    # Remove near-duplicate architectural proxies.
    deduped: List[Object2D] = []
    for obj in proxies:
        duplicate = False
        p = _poly_from_object(obj)
        for prev in deduped:
            if obj.label != prev.label:
                continue
            q = _poly_from_object(prev)
            inter = p.intersection(q).area if not p.is_empty and not q.is_empty else 0.0
            if inter / max(min(p.area, q.area), 1e-6) > 0.50:
                duplicate = True
                break
        if not duplicate:
            deduped.append(obj)
    for i, obj in enumerate(deduped, start=start_id):
        obj.id = i
        obj.object_id = i

    return deduped[: args.arch_max_total], {
        "enabled": True,
        "wall_points": int(len(wall_pts)),
        "raw_proxy_count": len(proxies),
        "deduped_proxy_count": len(deduped[: args.arch_max_total]),
        "edges": edge_debug,
    }


def sync_pseudo_labels_with_layout(pseudo: Dict[str, object], layout: SceneLayout) -> None:
    """Best-effort label synchronization for pseudo-ScanNet aggregation JSON.

    Architecture proxies do not have source vertices, so they are intentionally
    not inserted into segGroups.  Furniture objects keep their original pseudo
    segments with updated labels where possible.
    """
    seg_groups = pseudo.get("seg_groups", [])
    by_id = {obj.object_id: obj for obj in layout.objects if obj.label not in ARCHITECTURE_LABELS}
    for group in seg_groups:
        oid = int(group.get("objectId", group.get("id", -1)))
        if oid in by_id:
            group["label"] = by_id[oid].label


def build_layout_v4(args: argparse.Namespace) -> Tuple[SceneLayout, Dict[str, object], np.ndarray, Optional[np.ndarray], Dict[str, object]]:
    # Make v3 less likely to discard useful candidates before v4 post-processing.
    if args.v4_high_recall:
        args.filter_outside_room = False
        args.max_objects = max(args.max_objects, 120)
        args.min_object_points = min(args.min_object_points, 8)
        args.min_box_area = min(args.min_box_area, 0.002)
        args.wall_filter = args.wall_filter or "high_only"

    layout, pseudo, aligned, rgb, debug = v3.build_layout(args)
    room_poly = Polygon(layout.room_polygon).buffer(0)
    if room_poly.is_empty or not room_poly.is_valid:
        room_poly = MultiPoint(np.asarray(layout.room_polygon)).convex_hull

    # Step 1: upgrade semantic labels before fragment merging.
    refine_labels(layout, room_poly, preserve_proposal_labels=False)

    # Step 2: merge small/generic fragments into larger furniture candidates.
    merged_objects, merge_debug = merge_fragment_objects(
        layout.objects,
        room_poly=room_poly,
        merge_distance=args.fragment_merge_distance,
        min_merged_area=args.fragment_merge_min_area,
        max_merged_area_ratio=args.fragment_merge_max_area_ratio,
    )
    layout.objects = merged_objects

    # Step 3: enforce room boundary after merging/refinement.
    bounded_objects, boundary_debug = enforce_room_boundary(
        layout.objects,
        room_poly=room_poly,
        max_area_ratio=args.v4_max_box_area_ratio,
        outside_tolerance=args.v4_outside_tolerance,
        shrink_to_fit=args.v4_shrink_to_room,
        snap_centers=args.v4_snap_centers,
    )
    layout.objects = bounded_objects

    # Step 4: add architectural proxies (doors/windows) as non-source-vertex objects.
    arch_objects, arch_debug = detect_architecture_proxies(
        aligned,
        np.asarray(pseudo.get("wall_mask", np.zeros(len(aligned), dtype=bool))),
        room_poly,
        args,
        start_id=len(layout.objects) + 1,
    )
    layout.objects.extend(arch_objects)

    # Stable order: furniture first, architecture last.
    furniture = [o for o in layout.objects if o.label not in ARCHITECTURE_LABELS]
    architecture = [o for o in layout.objects if o.label in ARCHITECTURE_LABELS]
    furniture.sort(key=lambda o: (-(o.width * o.depth), o.label.lower(), o.id))
    architecture.sort(key=lambda o: (o.label, o.id))
    layout.objects = furniture + architecture
    for i, obj in enumerate(layout.objects, start=1):
        obj.id = i
        obj.object_id = i

    layout.room_bbox = polygon_bbox(room_poly)
    layout.metadata.update({
        "source": "user_ply_geometry_pseudo_scannet_v4",
        "v4_changes": [
            "generic proposal label refinement",
            "nearby fragment merging",
            "room-boundary snapping/shrinking/rejection",
            "heuristic door/window architectural proxies",
        ],
        "v4_fragment_merge": merge_debug,
        "v4_boundary_enforcement": boundary_debug,
        "v4_architecture_detection": arch_debug,
        "v4_warning": "Door/window and furniture labels are geometry-based proxies, not true semantic recognition.",
    })
    debug.update({
        "v4_fragment_merge": merge_debug,
        "v4_boundary_enforcement": boundary_debug,
        "v4_architecture_detection": arch_debug,
        "v4_object_labels": [o.label for o in layout.objects],
    })
    sync_pseudo_labels_with_layout(pseudo, layout)
    return layout, pseudo, aligned, rgb, debug


def write_debug_v4(out_dir: str | Path, layout: SceneLayout, aligned: np.ndarray, pseudo: Dict[str, object], debug: Dict[str, object], seed: int) -> None:
    # Reuse v3 debug images, then add a compact v4 architecture/boundary plot.
    v3.write_debug(out_dir, layout, aligned, pseudo, debug, seed)
    out_dir = Path(out_dir)
    room = np.asarray(layout.room_polygon, dtype=float)
    fig, ax = plt.subplots(figsize=(11, 11))
    if len(room) >= 3:
        closed = np.vstack([room, room[0]])
        ax.plot(closed[:, 0], closed[:, 1], linewidth=2.0, label="room polygon")
    for obj in layout.objects:
        corners = np.asarray(obj.footprint, dtype=float)
        if len(corners) < 3:
            continue
        closed = np.vstack([corners, corners[0]])
        lw = 2.2 if obj.label in ARCHITECTURE_LABELS else 1.2
        ax.plot(closed[:, 0], closed[:, 1], linewidth=lw)
        ax.text(obj.cx, obj.cy, f"{obj.id}:{obj.label}", fontsize=7)
    ax.axis("equal")
    ax.legend(loc="best")
    ax.set_title("v4 final layout: refined furniture + door/window proxies")
    fig.tight_layout()
    fig.savefig(out_dir / "debug_v4_final.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Raw user .ply -> pseudo-ScanNet + top-down draggable layout exporter v4")

    # Keep the v3 arguments so the two exporters can be swapped easily.
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
    p.add_argument("--room_resolution_m", type=float, default=0.04)
    p.add_argument("--room_close_iters", type=int, default=0)
    p.add_argument("--room_dilate_iters", type=int, default=0)
    p.add_argument("--room_simplify_m", type=float, default=0.004)
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

    p.add_argument("--occupancy_resolution", type=float, default=0.04)
    p.add_argument("--object_close_iters", type=int, default=0)
    p.add_argument("--object_dilate_iters", type=int, default=0)
    p.add_argument("--min_component_cells", type=int, default=1)
    p.add_argument("--min_object_points", type=int, default=8)
    p.add_argument("--min_object_height", type=float, default=0.025)
    p.add_argument("--max_object_height", type=float, default=2.40)
    p.add_argument("--min_box_area", type=float, default=0.002)
    p.add_argument("--max_box_area", type=float, default=8.0)
    p.add_argument("--max_box_area_ratio", type=float, default=0.35)
    p.add_argument("--max_box_long_side_ratio", type=float, default=0.90)
    p.add_argument("--box_padding", type=float, default=0.10)
    p.add_argument("--box_trim_percentile", type=float, default=0.5)
    p.add_argument("--filter_outside_room", action="store_true")
    p.add_argument("--outside_room_tolerance", type=float, default=0.30)

    p.add_argument("--nms_iou", type=float, default=0.50)
    p.add_argument("--nms_containment", type=float, default=0.80)
    p.add_argument("--max_objects", type=int, default=120)

    # v4-specific post-processing.
    p.add_argument("--v4_high_recall", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--fragment_merge_distance", type=float, default=0.28)
    p.add_argument("--fragment_merge_min_area", type=float, default=0.18)
    p.add_argument("--fragment_merge_max_area_ratio", type=float, default=0.28)
    p.add_argument("--v4_max_box_area_ratio", type=float, default=0.26)
    p.add_argument("--v4_outside_tolerance", type=float, default=0.12)
    p.add_argument("--v4_snap_centers", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--v4_shrink_to_room", action=argparse.BooleanOptionalAction, default=True)

    # Door/window proxy detection.
    p.add_argument("--detect_architecture", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--arch_wall_distance", type=float, default=0.18)
    p.add_argument("--arch_bin_width", type=float, default=0.16)
    p.add_argument("--arch_gap_fraction", type=float, default=0.18)
    p.add_argument("--arch_min_wall_len", type=float, default=1.20)
    p.add_argument("--arch_min_edge_points", type=int, default=20)
    p.add_argument("--arch_proxy_thickness", type=float, default=0.12)
    p.add_argument("--arch_door_min_width", type=float, default=0.55)
    p.add_argument("--arch_door_max_width", type=float, default=1.35)
    p.add_argument("--arch_door_max_z", type=float, default=2.10)
    p.add_argument("--arch_window_min_z", type=float, default=0.75)
    p.add_argument("--arch_window_max_z", type=float, default=1.85)
    p.add_argument("--arch_window_min_width", type=float, default=0.45)
    p.add_argument("--arch_window_max_width", type=float, default=2.40)
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

    layout, pseudo, aligned, rgb, debug = build_layout_v4(args)
    save_layout_json(layout, out_dir / "scene_layout.json")
    if not args.no_png:
        render_layout_png(layout, out_dir / "scene_layout.png")
    scene_id = args.scene_id or Path(args.ply).stem.replace(" ", "_")
    pseudo_dir = v3.write_pseudo_scannet_files(out_dir, scene_id, args.ply, aligned, rgb, pseudo, copy_original=not args.no_copy_original)
    if not args.no_debug:
        write_debug_v4(out_dir, layout, aligned, pseudo, debug, args.seed)

    print(f"Saved layout JSON to: {out_dir / 'scene_layout.json'}")
    if not args.no_png:
        print(f"Saved layout preview to: {out_dir / 'scene_layout.png'}")
    if not args.no_debug:
        print(f"Saved debug summary to: {out_dir / 'debug_detection_summary.json'}")
        print(f"Saved debug layers image to: {out_dir / 'debug_layers.png'}")
        print(f"Saved proposal debug image to: {out_dir / 'debug_proposals.png'}")
        print(f"Saved v4 final debug image to: {out_dir / 'debug_v4_final.png'}")
    print(f"Saved pseudo-ScanNet scene folder to: {pseudo_dir}")
    print(f"Objects exported: {len(layout.objects)}")


if __name__ == "__main__":
    main()
