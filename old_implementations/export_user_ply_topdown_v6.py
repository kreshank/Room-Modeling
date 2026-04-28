#!/usr/bin/env python3
"""
Raw user .ply -> pseudo-ScanNet + top-down draggable layout exporter (v6).

v6 is a post-processing and stabilization layer over v5.  It is designed for
real phone/LiDAR room scans where pure geometry produces these common problems:

  * a couch/bed/desk is broken into many small pieces;
  * small clutter on top of large furniture causes the whole item to be labeled
    as cabinet/shelf or tall_thin_object;
  * large furniture has a plausible relative location but the box rotation is
    not aligned with the room/walls;
  * some boxes drift slightly outside the room after rotation/repair;
  * doors/windows are absent because raw PLY files usually do not contain clean
    semantic opening annotations.

Important: this is still geometry-only pseudo-recognition.  v6 makes the output
more editable and more plausible, but true object recognition still needs a
ScanNet-trained 3D instance segmentation model or a user correction step.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
import numpy as np
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.ops import unary_union

from export_scannet_topdown import Object2D, SceneLayout, polygon_bbox, render_layout_png, save_layout_json
import export_user_ply_topdown_v3 as v3
import export_user_ply_topdown_v5 as v5

ARCHITECTURE_LABELS = set(v5.ARCHITECTURE_LABELS) | {"door_candidate", "window_candidate"}
WEAK_FRAGMENT_LABELS = {
    "small_object", "small_object_proposal", "furniture_proposal", "unknown_object",
    "unknown_furniture", "large_furniture", "small_furniture", "chair_or_small_furniture",
    "table_or_desk", "bed_or_sofa", "floor_lamp_or_tall_thin_object",
}
SURFACE_LABELS = {"bed", "sofa", "table", "desk", "work_desk", "kitchen_counter"}
LARGE_ALIGN_LABELS = SURFACE_LABELS | {"cabinet_or_shelf", "shelf_or_bookcase"}


def angle_norm(theta: float) -> float:
    """Normalize angle to [-pi, pi)."""
    return float((theta + math.pi) % (2 * math.pi) - math.pi)


def angle_diff(a: float, b: float) -> float:
    return abs(angle_norm(a - b))


def object_area(obj: Object2D) -> float:
    return float(max(obj.width, 0.0) * max(obj.depth, 0.0))


def object_long_short(obj: Object2D) -> Tuple[float, float]:
    return float(max(obj.width, obj.depth)), float(min(obj.width, obj.depth))


def object_poly(obj: Object2D) -> Polygon:
    return v5.object_poly(obj)


def recompute(obj: Object2D) -> None:
    v5.recompute_footprint(obj)


def room_edges(room_poly: Polygon) -> List[Tuple[np.ndarray, np.ndarray, float, float]]:
    if room_poly.is_empty or len(room_poly.exterior.coords) < 2:
        return []
    coords = np.asarray(room_poly.exterior.coords[:-1], dtype=float)
    edges = []
    for i in range(len(coords)):
        p0 = coords[i]
        p1 = coords[(i + 1) % len(coords)]
        length = float(np.linalg.norm(p1 - p0))
        if length < 1e-6:
            continue
        theta = float(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
        edges.append((p0, p1, length, theta))
    return edges


def nearest_wall_orientation(room_poly: Polygon, x: float, y: float) -> Tuple[Optional[float], float]:
    edges = room_edges(room_poly)
    if not edges:
        return None, float("inf")
    p = np.array([x, y], dtype=float)
    best_theta: Optional[float] = None
    best_dist = float("inf")
    for p0, p1, length, theta in edges:
        v = p1 - p0
        t = float(np.clip(((p - p0) @ v) / max(float(v @ v), 1e-9), 0.0, 1.0))
        q = p0 + t * v
        dist = float(np.linalg.norm(p - q))
        if dist < best_dist:
            best_dist = dist
            best_theta = theta
    return best_theta, best_dist


def snap_theta_to_wall(obj: Object2D, room_poly: Polygon, max_distance: float) -> Tuple[bool, str]:
    wall_theta, dist = nearest_wall_orientation(room_poly, obj.cx, obj.cy)
    if wall_theta is None or dist > max_distance:
        return False, "no_near_wall"
    candidates = [wall_theta, wall_theta + math.pi / 2.0, wall_theta + math.pi, wall_theta - math.pi / 2.0]
    new_theta = min(candidates, key=lambda t: angle_diff(obj.theta, t))
    old = obj.theta
    obj.theta = angle_norm(new_theta)
    recompute(obj)
    return True, f"snapped_from_{old:.3f}_to_{obj.theta:.3f}_wall_dist_{dist:.2f}"


def oriented_box_from_objects(objs: Sequence[Object2D], padding: float = 0.05) -> Tuple[np.ndarray, float, float, float, np.ndarray]:
    pts: List[List[float]] = []
    for o in objs:
        if o.footprint:
            pts.extend([[float(x), float(y)] for x, y in o.footprint])
        pts.append([float(o.cx), float(o.cy)])
    arr = np.asarray(pts, dtype=float)
    if len(arr) < 3:
        c = arr.mean(axis=0) if len(arr) else np.zeros(2)
        return c, 0.2, 0.2, 0.0, np.asarray(v5.bbox_corners(c[0], c[1], 0.2, 0.2, 0.0))
    c, w, d, theta, corners = v3.robust_box_from_points(arr, trim_pct=0.0)
    w = float(max(w + 2 * padding, 0.05))
    d = float(max(d + 2 * padding, 0.05))
    corners = np.asarray(v5.bbox_corners(float(c[0]), float(c[1]), w, d, float(theta)), dtype=float)
    return c, w, d, float(theta), corners


def classify_geometry_v6(obj: Object2D, room_poly: Polygon, source_labels: Optional[Sequence[str]] = None) -> Tuple[str, float, str]:
    """More assertive geometry labeler.

    Main changes from v5:
      * footprint dominates over z_max, so clutter on top does not turn a bed/table
        into a cabinet/shelf;
      * floor_lamp is only used for genuinely tiny footprints;
      * cabinet/shelf is a late fallback unless the object is tall and narrow.
    """
    if obj.label in ARCHITECTURE_LABELS:
        return obj.label, 0.80, "preserved_architecture"

    w = float(max(obj.width, 1e-6))
    d = float(max(obj.depth, 1e-6))
    h = float(max(obj.height, obj.z_max - obj.z_min, 0.0))
    zmax = float(obj.z_max)
    area = w * d
    long_side = max(w, d)
    short_side = min(w, d)
    aspect = long_side / max(short_side, 1e-6)
    wall_theta, wall_dist = nearest_wall_orientation(room_poly, obj.cx, obj.cy)
    near_wall = wall_dist <= 0.65
    old = (obj.label or "").lower()
    src = " ".join(source_labels or [old, obj.raw_label or ""]).lower()

    # Only tiny footprint objects can be lamps/stands.  This avoids couch/bed
    # fragments getting over-labeled as floor_lamp_or_tall_thin_object.
    if area <= 0.075 and long_side <= 0.42 and zmax >= 0.90:
        return "floor_lamp_or_tall_thin_object", 0.46, "tiny_footprint_tall_object"

    # Large low/deep objects.  This rule intentionally tolerates zmax up to 1.55
    # because phone scans often include pillows, blankets, monitor arms, or small
    # clutter sitting on top.
    if long_side >= 1.65 and short_side >= 0.72 and area >= 1.20 and zmax <= 1.65:
        if short_side >= 0.88 and area >= 1.45 and aspect <= 3.1:
            return "bed", 0.62, "large_deep_rectangular_footprint_clutter_tolerant"
        return "sofa", 0.55, "large_low_long_seating_footprint_clutter_tolerant"

    # Partial sofa/bed regions: common when the scan captures only cushions/back.
    if long_side >= 1.20 and short_side >= 0.45 and area >= 0.62 and zmax <= 1.65:
        if near_wall and short_side >= 0.70 and area >= 1.00:
            return "bed", 0.44, "partial_deep_wall_adjacent_bed_like"
        return "sofa", 0.43, "partial_large_low_sofa_like"

    # Work surfaces.  Table/desk/counter checks come before cabinet/shelf.
    table_like_height = zmax <= 1.45 and h <= 1.45
    if table_like_height and 0.16 <= area <= 3.40 and long_side >= 0.55 and short_side >= 0.24:
        if near_wall and long_side >= 1.10 and aspect >= 1.25:
            if long_side >= 1.35 and short_side <= 0.95 and zmax >= 0.55:
                # Wall-adjacent, long, table/counter height.  Counter is chosen
                # only when it is especially long/narrow or old label indicated it.
                if long_side >= 1.70 and short_side <= 0.75:
                    return "kitchen_counter", 0.51, "long_narrow_wall_adjacent_counter_surface"
                return "work_desk", 0.50, "wall_adjacent_work_surface"
            return "desk", 0.45, "wall_adjacent_table_height_surface"
        if area >= 0.25:
            return "table", 0.48, "free_or_central_table_surface"

    # Chair/stool-like.
    if 0.06 <= area <= 0.55 and long_side <= 0.95 and zmax <= 1.35:
        return "chair_or_stool", 0.34, "small_sitting_height_footprint"

    # Tall storage is only chosen when footprint/height really suggest it, and
    # only after surface/bed/sofa/table possibilities have failed.
    if zmax >= 1.25 and h >= 0.80 and area >= 0.10:
        if near_wall and (short_side <= 0.70 or aspect >= 1.30):
            return "cabinet_or_shelf", 0.39, "tall_narrow_wall_storage_like"
        if area >= 0.25 and aspect >= 1.25:
            return "shelf_or_bookcase", 0.35, "tall_rectangular_storage_like"

    if area >= 0.30:
        if near_wall:
            return "desk", 0.27, "generic_medium_wall_adjacent_surface"
        return "table", 0.26, "generic_medium_surface"
    return "small_object", 0.16, "too_small_or_sparse_for_specific_label"


def make_merged_object(obj_id: int, objs: Sequence[Object2D], room_poly: Polygon, padding: float) -> Object2D:
    center, w, d, theta, corners = oriented_box_from_objects(objs, padding=padding)
    zmin = float(min(o.z_min for o in objs))
    zmax = float(max(o.z_max for o in objs))
    h = float(max(0.0, zmax - zmin))
    merged = Object2D(
        id=obj_id,
        object_id=obj_id,
        label="furniture_proposal",
        raw_label="v6_merged_from__" + ";".join(f"{o.id}:{o.label}" for o in objs),
        cx=float(center[0]),
        cy=float(center[1]),
        width=float(w),
        depth=float(d),
        theta=float(theta),
        z_min=zmin,
        z_max=zmax,
        height=h,
        point_count=int(sum(max(0, o.point_count) for o in objs)),
        footprint=[[float(x), float(y)] for x, y in corners.tolist()],
        bbox3d_center=[float(center[0]), float(center[1]), float((zmin + zmax) / 2.0)],
        bbox3d_size=[float(w), float(d), h],
    )
    label, conf, reason = classify_geometry_v6(merged, room_poly, [o.label for o in objs])
    merged.label = label
    merged.raw_label += f"__v6_refined_to__{label}__{reason}__conf_{conf:.2f}"
    return merged


def should_absorb_small_into_large(small: Object2D, large: Object2D, room_poly: Polygon, args: argparse.Namespace) -> bool:
    if small.label in ARCHITECTURE_LABELS or large.label in ARCHITECTURE_LABELS:
        return False
    small_area = object_area(small)
    large_area = max(object_area(large), 1e-6)
    if large.label not in SURFACE_LABELS and large_area < args.v6_large_surface_min_area:
        return False
    if small_area > min(args.v6_clutter_max_area, large_area * args.v6_clutter_area_ratio):
        return False
    sp = object_poly(small)
    lp = object_poly(large).buffer(args.v6_clutter_buffer)
    if sp.is_empty or lp.is_empty:
        return False
    if lp.contains(Point(small.cx, small.cy)) or lp.intersects(sp):
        return True
    return False


def suppress_surface_clutter(objects: List[Object2D], room_poly: Polygon, args: argparse.Namespace) -> Tuple[List[Object2D], Dict[str, object]]:
    if not args.v6_absorb_surface_clutter:
        return objects, {"enabled": False}
    objs = list(objects)
    suppress: set[int] = set()
    events = []
    # Larger objects first.
    large_order = sorted(range(len(objs)), key=lambda i: object_area(objs[i]), reverse=True)
    for li in large_order:
        large = objs[li]
        if li in suppress or large.label in ARCHITECTURE_LABELS:
            continue
        for si, small in enumerate(objs):
            if si == li or si in suppress:
                continue
            if should_absorb_small_into_large(small, large, room_poly, args):
                suppress.add(si)
                large.raw_label += f"__absorbed_surface_clutter_{small.id}:{small.label}"
                large.point_count += max(0, small.point_count)
                events.append({"absorbed_id": small.id, "absorbed_label": small.label, "into_id": large.id, "into_label": large.label})
    kept = [o for i, o in enumerate(objs) if i not in suppress]
    return kept, {"enabled": True, "absorbed_count": len(suppress), "events": events}


def cluster_fragment_indices(objects: List[Object2D], room_poly: Polygon, args: argparse.Namespace) -> List[List[int]]:
    n = len(objects)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        a = objects[i]
        if a.label in ARCHITECTURE_LABELS:
            continue
        for j in range(i + 1, n):
            b = objects[j]
            if b.label in ARCHITECTURE_LABELS:
                continue
            # Only aggressively merge weak/small fragments. Strong objects can
            # absorb clutter separately, but should not merge with each other.
            weak_pair = (a.label in WEAK_FRAGMENT_LABELS or object_area(a) <= args.v6_fragment_max_area) and \
                        (b.label in WEAK_FRAGMENT_LABELS or object_area(b) <= args.v6_fragment_max_area)
            if not weak_pair:
                continue
            pa, pb = object_poly(a), object_poly(b)
            if pa.is_empty or pb.is_empty:
                continue
            dist = float(pa.distance(pb))
            if dist > args.v6_fragment_merge_distance:
                continue
            center_dist = math.hypot(a.cx - b.cx, a.cy - b.cy)
            if center_dist > args.v6_fragment_center_distance:
                continue
            # Do not merge if the combined footprint would be room-sized.
            c, w, d, theta, corners = oriented_box_from_objects([a, b], padding=args.v6_merge_padding)
            merged_area = w * d
            room_area = max(float(room_poly.area), 1e-6)
            if merged_area > args.v6_merge_max_area or merged_area / room_area > args.v6_merge_max_room_ratio:
                continue
            union(i, j)

    clusters: Dict[int, List[int]] = {}
    for i in range(n):
        clusters.setdefault(find(i), []).append(i)
    return list(clusters.values())


def merge_fragments(objects: List[Object2D], room_poly: Polygon, args: argparse.Namespace) -> Tuple[List[Object2D], Dict[str, object]]:
    if not args.v6_merge_fragments:
        return objects, {"enabled": False}
    clusters = cluster_fragment_indices(objects, room_poly, args)
    new_objects: List[Object2D] = []
    events = []
    next_id = 1
    for inds in clusters:
        group = [objects[i] for i in inds]
        if len(group) == 1:
            o = group[0]
            o.id = next_id
            o.object_id = next_id
            new_objects.append(o)
            next_id += 1
            continue
        merged = make_merged_object(next_id, group, room_poly, args.v6_merge_padding)
        area = object_area(merged)
        if area < args.v6_merge_min_output_area or merged.label in {"small_object", "floor_lamp_or_tall_thin_object"}:
            # Cluster is just nearby clutter; keep individual pieces.
            for o in group:
                o.id = next_id
                o.object_id = next_id
                new_objects.append(o)
                next_id += 1
            continue
        new_objects.append(merged)
        events.append({"merged_ids": [o.id for o in group], "new_id": merged.id, "new_label": merged.label, "new_area": area})
        next_id += 1
    return new_objects, {"enabled": True, "cluster_count": len(clusters), "merged_count": len(events), "events": events}


def relabel_and_snap(objects: List[Object2D], room_poly: Polygon, args: argparse.Namespace) -> Dict[str, object]:
    events = []
    for obj in objects:
        old = obj.label
        new, conf, reason = classify_geometry_v6(obj, room_poly)
        obj.label = new
        obj.raw_label = f"{obj.raw_label}__v6_refined_from__{old}__to__{new}__{reason}__conf_{conf:.2f}"
        snap_event = None
        if args.v6_snap_to_walls and obj.label in LARGE_ALIGN_LABELS and object_area(obj) >= args.v6_snap_min_area:
            snapped, snap_event = snap_theta_to_wall(obj, room_poly, args.v6_snap_wall_distance)
        events.append({"id": obj.id, "old": old, "new": new, "confidence": conf, "reason": reason, "snap": snap_event})
    return {"events": events}


def light_repair_positions(objects: List[Object2D], room_poly: Polygon, args: argparse.Namespace) -> Dict[str, object]:
    events = []
    if room_poly.is_empty:
        return {"enabled": False, "reason": "empty_room"}
    rep = room_poly.representative_point()
    target = np.array([rep.x, rep.y], dtype=float)
    kept: List[Object2D] = []
    for obj in objects:
        if obj.label in ARCHITECTURE_LABELS:
            kept.append(obj)
            continue
        ratio = v5.area_ratio_outside(obj, room_poly, args.v6_room_tolerance)
        center_ok = room_poly.buffer(args.v6_room_tolerance).contains(Point(obj.cx, obj.cy))
        action = "preserve"
        if ratio > args.v6_max_outside_ratio or not center_ok:
            if args.v6_position_repair == "reject":
                action = "reject_outside"
                events.append({"id": obj.id, "label": obj.label, "action": action, "outside_ratio": ratio})
                continue
            elif args.v6_position_repair == "nudge":
                start = np.array([obj.cx, obj.cy], dtype=float)
                vec = target - start
                norm = float(np.linalg.norm(vec))
                if norm > 1e-8:
                    vec /= norm
                    best = (ratio, obj.cx, obj.cy)
                    for step in np.linspace(0.0, args.v6_max_nudge_m, 18):
                        obj.cx = float(start[0] + vec[0] * step)
                        obj.cy = float(start[1] + vec[1] * step)
                        recompute(obj)
                        r = v5.area_ratio_outside(obj, room_poly, args.v6_room_tolerance)
                        if r < best[0]:
                            best = (r, obj.cx, obj.cy)
                    obj.cx = best[1]
                    obj.cy = best[2]
                    recompute(obj)
                    action = "light_nudge" if best[0] < ratio else "nudge_no_improvement"
        kept.append(obj)
        events.append({"id": obj.id, "label": obj.label, "action": action, "outside_ratio_before": ratio, "outside_ratio_after": v5.area_ratio_outside(obj, room_poly, args.v6_room_tolerance)})
    objects[:] = kept
    return {"enabled": True, "policy": args.v6_position_repair, "events": events}


def add_architecture_fallbacks(layout: SceneLayout, room_poly: Polygon, args: argparse.Namespace) -> Dict[str, object]:
    if not args.v6_architecture_fallbacks or room_poly.is_empty:
        return {"enabled": False}
    existing_doors = [o for o in layout.objects if o.label in {"door", "door_candidate"}]
    existing_windows = [o for o in layout.objects if o.label in {"window", "window_candidate"}]
    edges = sorted(room_edges(room_poly), key=lambda e: e[2], reverse=True)
    room_center = np.array([room_poly.representative_point().x, room_poly.representative_point().y], dtype=float)
    added = []
    next_id = max([o.id for o in layout.objects], default=0) + 1

    def make_on_edge(label: str, edge, frac: float, length: float, zmin: float, zmax: float, reason: str) -> Object2D:
        nonlocal next_id
        p0, p1, edge_len, theta = edge
        center = p0 + (p1 - p0) * frac
        inward = room_center - center
        n = float(np.linalg.norm(inward))
        if n > 1e-9:
            center = center + inward / n * (args.arch_proxy_thickness * 0.9)
        obj = v5.make_arch_obj(next_id, label, center, min(length, edge_len * 0.55), args.arch_proxy_thickness, theta, zmin, zmax, f"v6_fallback_{reason}")
        next_id += 1
        return obj

    if len(existing_doors) < args.v6_min_door_proxies and edges:
        # Place a candidate on the longest edge away from corners.
        for edge in edges[: max(1, args.v6_arch_fallback_edges)]:
            if len(existing_doors) + sum(1 for a in added if a.label.startswith("door")) >= args.v6_min_door_proxies:
                break
            obj = make_on_edge("door_candidate", edge, 0.18 if len(added) % 2 == 0 else 0.82, args.arch_door_min_width * 1.4, 0.0, args.arch_door_max_z, "door_on_long_wall_needs_confirmation")
            layout.objects.append(obj)
            added.append({"id": obj.id, "label": obj.label, "reason": obj.raw_label})

    if len(existing_windows) < args.v6_min_window_proxies and edges:
        # Add candidates on long walls, biased toward wall centers.  These are
        # deliberately candidates because raw PLY scans often do not preserve
        # transparent glass/opening evidence.
        count_needed = args.v6_min_window_proxies - len(existing_windows)
        for k, edge in enumerate(edges[: max(1, args.v6_arch_fallback_edges)]):
            if count_needed <= 0:
                break
            frac = 0.50 if k == 0 else 0.35 if k % 2 else 0.65
            obj = make_on_edge("window_candidate", edge, frac, min(args.arch_window_max_width, max(args.arch_window_min_width, edge[2] * 0.35)), args.arch_window_min_z, args.arch_window_max_z, "window_on_long_wall_needs_confirmation")
            layout.objects.append(obj)
            added.append({"id": obj.id, "label": obj.label, "reason": obj.raw_label})
            count_needed -= 1
    return {"enabled": True, "added": added, "existing_doors": len(existing_doors), "existing_windows": len(existing_windows)}


def rebuild_pseudo_from_layout(layout: SceneLayout, pseudo: Dict[str, object], aligned: np.ndarray, args: argparse.Namespace) -> Dict[str, object]:
    seg_indices = np.zeros(len(aligned), dtype=np.int32)
    semantic_labels = np.full(len(aligned), v3.PSEUDO_SEMANTIC_ID["unlabeled"], dtype=np.int32)
    floor_mask = np.asarray(pseudo.get("floor_mask", np.zeros(len(aligned), dtype=bool)), dtype=bool)
    wall_mask = np.asarray(pseudo.get("wall_mask", np.zeros(len(aligned), dtype=bool)), dtype=bool)
    ceiling_mask = np.asarray(pseudo.get("ceiling_mask", np.zeros(len(aligned), dtype=bool)), dtype=bool)
    seg_indices[floor_mask] = 0
    semantic_labels[floor_mask] = v3.PSEUDO_SEMANTIC_ID.get("floor", 1)
    seg_indices[wall_mask] = 1
    semantic_labels[wall_mask] = v3.PSEUDO_SEMANTIC_ID.get("wall", 2)
    seg_indices[ceiling_mask] = 2
    semantic_labels[ceiling_mask] = v3.PSEUDO_SEMANTIC_ID.get("ceiling", 22)

    seg_groups = []
    xy = aligned[:, :2]
    z = aligned[:, 2]
    next_seg = 10
    for obj in layout.objects:
        if not obj.footprint or obj.label in ARCHITECTURE_LABELS:
            continue
        poly_pts = np.asarray(obj.footprint, dtype=float)
        if len(poly_pts) < 3:
            continue
        path = MplPath(poly_pts)
        bbox = np.array([poly_pts[:, 0].min(), poly_pts[:, 1].min(), poly_pts[:, 0].max(), poly_pts[:, 1].max()])
        bbox_mask = (xy[:, 0] >= bbox[0]) & (xy[:, 0] <= bbox[2]) & (xy[:, 1] >= bbox[1]) & (xy[:, 1] <= bbox[3])
        z_mask = (z >= obj.z_min - args.v6_relabel_z_tolerance) & (z <= obj.z_max + args.v6_relabel_z_tolerance)
        cand = np.where(bbox_mask & z_mask)[0]
        if len(cand):
            inside = path.contains_points(xy[cand])
            ids = cand[inside]
        else:
            ids = np.array([], dtype=int)
        if len(ids) == 0:
            continue
        sem_id = v3.PSEUDO_SEMANTIC_ID.get(obj.label, v3.PSEUDO_SEMANTIC_ID.get("unknown_object", 0))
        seg_indices[ids] = next_seg
        semantic_labels[ids] = sem_id
        seg_groups.append({"objectId": int(obj.object_id), "id": int(obj.object_id), "label": obj.label, "segments": [int(next_seg)]})
        next_seg += 1
    pseudo["seg_indices"] = seg_indices
    pseudo["semantic_labels"] = semantic_labels
    pseudo["seg_groups"] = seg_groups
    return {"seg_group_count": len(seg_groups), "note": "rebuilt pseudo-ScanNet segments from final v6 layout footprints"}


def build_layout_v6(args: argparse.Namespace):
    layout, pseudo, aligned, rgb, debug = v5.build_layout_v5(args)
    room_poly = Polygon(layout.room_polygon).buffer(0)
    if room_poly.is_empty or not room_poly.is_valid:
        room_poly = MultiPoint(np.asarray(layout.room_polygon, dtype=float)).convex_hull

    # Split architecture and furniture.  Reprocess only furniture.
    architecture = [o for o in layout.objects if o.label in ARCHITECTURE_LABELS]
    furniture = [o for o in layout.objects if o.label not in ARCHITECTURE_LABELS]

    # First relabel individual proposals with the stricter v6 priority order.
    relabel_debug_1 = relabel_and_snap(furniture, room_poly, args)

    # Merge small fragments into larger plausible furniture candidates.
    furniture, merge_debug = merge_fragments(furniture, room_poly, args)

    # Re-label after merging, then absorb clutter lying on top of large surfaces.
    relabel_debug_2 = relabel_and_snap(furniture, room_poly, args)
    furniture, clutter_debug = suppress_surface_clutter(furniture, room_poly, args)

    # Final relabel after clutter suppression and light room repair.
    relabel_debug_3 = relabel_and_snap(furniture, room_poly, args)
    position_debug = light_repair_positions(furniture, room_poly, args)

    layout.objects = furniture + architecture
    arch_fallback_debug = add_architecture_fallbacks(layout, room_poly, args)

    # Stable spatial ordering; keep architecture after furniture for editor list.
    furniture = [o for o in layout.objects if o.label not in ARCHITECTURE_LABELS]
    architecture = [o for o in layout.objects if o.label in ARCHITECTURE_LABELS]
    furniture.sort(key=lambda o: (round(o.cy, 3), round(o.cx, 3), o.label.lower()))
    architecture.sort(key=lambda o: (o.label, round(o.cy, 3), round(o.cx, 3)))
    layout.objects = furniture + architecture
    for i, obj in enumerate(layout.objects, start=1):
        obj.id = i
        obj.object_id = i

    pseudo_rebuild_debug = rebuild_pseudo_from_layout(layout, pseudo, aligned, args)

    layout.room_bbox = polygon_bbox(room_poly)
    layout.metadata.update({
        "source": "user_ply_geometry_pseudo_scannet_v6",
        "v6_changes": [
            "fragment merging for over-segmented couches/beds/desks",
            "clutter-tolerant label priority: bed/sofa/table/desk before cabinet/shelf",
            "floor_lamp label restricted to genuinely tiny tall footprints",
            "optional wall-aligned rotations for large furniture",
            "light position repair without nearest-wall snapping",
            "fallback door/window candidates when opening evidence is absent",
            "pseudo-ScanNet segments rebuilt from final v6 footprints",
        ],
        "v6_warning": "This is still geometry-only pseudo-recognition. Non-stereotypical furniture and doors/windows often require manual correction or ML segmentation.",
        "v6_fragment_merge": merge_debug,
        "v6_clutter_suppression": clutter_debug,
        "v6_relabel_before_merge": relabel_debug_1,
        "v6_relabel_after_merge": relabel_debug_2,
        "v6_relabel_final": relabel_debug_3,
        "v6_position_repair": position_debug,
        "v6_architecture_fallbacks": arch_fallback_debug,
        "v6_pseudo_rebuild": pseudo_rebuild_debug,
    })
    debug.update({
        "v6_fragment_merge": merge_debug,
        "v6_clutter_suppression": clutter_debug,
        "v6_relabel_before_merge": relabel_debug_1,
        "v6_relabel_after_merge": relabel_debug_2,
        "v6_relabel_final": relabel_debug_3,
        "v6_position_repair": position_debug,
        "v6_architecture_fallbacks": arch_fallback_debug,
        "v6_pseudo_rebuild": pseudo_rebuild_debug,
        "v6_final_labels": [o.label for o in layout.objects],
    })
    return layout, pseudo, aligned, rgb, debug


def write_debug_v6(out_dir: str | Path, layout: SceneLayout, aligned: np.ndarray, pseudo: Dict[str, object], debug: Dict[str, object], seed: int) -> None:
    # Reuse v5/v3 debug products, then add a v6-specific overlay.
    v5.write_debug_v5(out_dir, layout, aligned, pseudo, debug, seed)
    out_dir = Path(out_dir)
    room = np.asarray(layout.room_polygon, dtype=float)
    fig, ax = plt.subplots(figsize=(12, 12))
    # Sample for readability.
    rng = np.random.default_rng(seed)
    n = len(aligned)
    idx = rng.choice(n, size=min(n, 180000), replace=False) if n > 180000 else np.arange(n)
    pts = aligned[idx]
    ax.scatter(pts[:, 0], pts[:, 1], s=0.20, alpha=0.08, label="aligned raw points")
    if len(room) >= 3:
        closed = np.vstack([room, room[0]])
        ax.plot(closed[:, 0], closed[:, 1], linewidth=2.3, label="room polygon")
    for obj in layout.objects:
        corners = np.asarray(obj.footprint, dtype=float)
        if len(corners) < 3:
            continue
        closed = np.vstack([corners, corners[0]])
        lw = 2.5 if obj.label in ARCHITECTURE_LABELS else 1.4
        ax.plot(closed[:, 0], closed[:, 1], linewidth=lw)
        ax.scatter([obj.cx], [obj.cy], s=10)
        ax.text(obj.cx, obj.cy, f"{obj.id}:{obj.label}", fontsize=7)
    ax.axis("equal")
    ax.legend(loc="best", markerscale=8)
    ax.set_title("v6 final: merged fragments, wall-aligned furniture, architecture candidates")
    fig.tight_layout()
    fig.savefig(out_dir / "debug_v6_final_overlay.png", dpi=180)
    plt.close(fig)
    with (out_dir / "debug_v6_summary.json").open("w", encoding="utf-8") as f:
        json.dump(debug, f, indent=2)


def parse_args() -> argparse.Namespace:
    # Parse v6-only arguments first, then let v5 parse all existing pipeline args.
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--v6_merge_fragments", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--v6_fragment_merge_distance", type=float, default=0.34)
    p.add_argument("--v6_fragment_center_distance", type=float, default=0.95)
    p.add_argument("--v6_fragment_max_area", type=float, default=0.22)
    p.add_argument("--v6_merge_padding", type=float, default=0.08)
    p.add_argument("--v6_merge_min_output_area", type=float, default=0.38)
    p.add_argument("--v6_merge_max_area", type=float, default=3.80)
    p.add_argument("--v6_merge_max_room_ratio", type=float, default=0.22)

    p.add_argument("--v6_absorb_surface_clutter", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--v6_clutter_max_area", type=float, default=0.16)
    p.add_argument("--v6_clutter_area_ratio", type=float, default=0.16)
    p.add_argument("--v6_clutter_buffer", type=float, default=0.10)
    p.add_argument("--v6_large_surface_min_area", type=float, default=0.55)

    p.add_argument("--v6_snap_to_walls", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--v6_snap_min_area", type=float, default=0.35)
    p.add_argument("--v6_snap_wall_distance", type=float, default=0.80)

    p.add_argument("--v6_position_repair", choices=["preserve", "nudge", "reject"], default="nudge")
    p.add_argument("--v6_room_tolerance", type=float, default=0.10)
    p.add_argument("--v6_max_outside_ratio", type=float, default=0.32)
    p.add_argument("--v6_max_nudge_m", type=float, default=0.35)

    p.add_argument("--v6_architecture_fallbacks", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--v6_min_door_proxies", type=int, default=1)
    p.add_argument("--v6_min_window_proxies", type=int, default=2)
    p.add_argument("--v6_arch_fallback_edges", type=int, default=3)
    p.add_argument("--v6_relabel_z_tolerance", type=float, default=0.08)

    v6_args, remaining = p.parse_known_args()
    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0]] + remaining
        args = v5.parse_args()
    finally:
        sys.argv = old_argv
    for k, v in vars(v6_args).items():
        setattr(args, k, v)
    return args


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layout, pseudo, aligned, rgb, debug = build_layout_v6(args)
    save_layout_json(layout, out_dir / "scene_layout.json")
    if not args.no_png:
        render_layout_png(layout, out_dir / "scene_layout.png")
    scene_id = args.scene_id or Path(args.ply).stem.replace(" ", "_")
    pseudo_dir = v3.write_pseudo_scannet_files(out_dir, scene_id, args.ply, aligned, rgb, pseudo, copy_original=not args.no_copy_original)
    if not args.no_debug:
        write_debug_v6(out_dir, layout, aligned, pseudo, debug, args.seed)

    print(f"Saved layout JSON to: {out_dir / 'scene_layout.json'}")
    if not args.no_png:
        print(f"Saved layout preview to: {out_dir / 'scene_layout.png'}")
    if not args.no_debug:
        print(f"Saved debug summary to: {out_dir / 'debug_detection_summary.json'}")
        print(f"Saved v6 debug summary to: {out_dir / 'debug_v6_summary.json'}")
        print(f"Saved v6 final overlay to: {out_dir / 'debug_v6_final_overlay.png'}")
    print(f"Saved pseudo-ScanNet scene folder to: {pseudo_dir}")
    print(f"Objects exported: {len(layout.objects)}")


if __name__ == "__main__":
    main()
