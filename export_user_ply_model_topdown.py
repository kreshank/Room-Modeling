#!/usr/bin/env python3
"""
Model-ready user .ply -> labeled top-down layout exporter.

This is the next-stage exporter after the geometry-only v1-v5 pipeline.  It keeps
`.ply` as the only user-provided room input, but assumes the semantic/object
recognition step comes from a real 3D segmentation model (Mask3D/Pointcept/
OpenMask3D-style) or an external script that writes model predictions.

Why this file exists
--------------------
The geometry-only exporters are useful baselines, but they cannot reliably tell a
bed from a sofa/cabinet/table in arbitrary raw scans.  This script moves the
project architecture to the intended model-backed shape:

    user .ply
      -> optional external model inference
      -> predicted 3D instance masks + labels
      -> robust mask-based boxes
      -> room/wall-axis orientation snapping
      -> same scene_layout.json used by room_editor.html
      -> pseudo-ScanNet files for compatibility

The script does NOT ship pretrained model weights.  Instead, it defines the
stable interface between your model and the existing ScanNet/top-down/editor
pipeline.  You can use it in three ways:

1. Convert an existing predictions JSON:

    python export_user_ply_model_topdown.py \
      --ply data/user_scans/my_room.ply \
      --predictions_json out/model_predictions.json \
      --out_dir out/my_room_model \
      --scene_id my_room

2. Run an external inference command first, then convert its output:

    python export_user_ply_model_topdown.py \
      --ply data/user_scans/my_room.ply \
      --inference_cmd "python run_my_model.py --ply {ply} --out {predictions}" \
      --out_dir out/my_room_model \
      --scene_id my_room

3. Use it as a target adapter for Mask3D/OpenMask3D/Pointcept outputs by writing
   the simple predictions schema documented below.

Predictions JSON schema
-----------------------
Minimal instance-mask form:

{
  "coordinate_frame": "input",            // "input" or "aligned"; point indices work in either frame
  "instances": [
    {"id": 1, "label": "bed", "score": 0.91, "point_indices": [0, 4, 9, ...]},
    {"id": 2, "class_name": "couch", "confidence": 0.82, "indices": [12, 13, ...]}
  ],
  "semantic_regions": [
    {"label": "floor", "point_indices": [1, 2, 3, ...]},
    {"label": "wall",  "point_indices": [8, 9, 10, ...]}
  ]
}

Alternative bbox form, useful when your detector predicts boxes directly:

{
  "coordinate_frame": "aligned",
  "instances": [
    {
      "label": "desk",
      "score": 0.74,
      "bbox": {"cx": 1.2, "cy": 3.4, "width": 1.4, "depth": 0.7,
               "theta": 1.57, "z_min": 0.0, "z_max": 0.9}
    }
  ]
}

Point-index masks are strongly preferred over precomputed boxes because the
exporter can robustly recompute wall-aligned top-down boxes from the actual mask.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import MultiPoint, Point, Polygon
from shapely.ops import unary_union

from export_scannet_topdown import (
    Object2D,
    SceneLayout,
    polygon_bbox,
    polygon_to_coords,
    render_layout_png,
    save_layout_json,
)

# Reuse stable geometry helpers: PLY parsing, floor alignment, room polygon estimation,
# pseudo-ScanNet writing, and debug writing.
import user_ply_geometry_common as geom


TARGET_LABELS = [
    "wall", "window", "door", "sofa", "bed", "drawer", "table", "desk", "work_desk",
    "chair", "floor_lamp", "box", "container", "kitchen_counter", "stove",
    "cabinet_or_shelf", "shelf_or_bookcase", "unknown_object",
]

ARCHITECTURE_LABELS = {"wall", "window", "door"}
STRUCTURAL_LABELS = {"floor", "ceiling", "wall"}
NON_OBJECT_STRUCTURAL = {"floor", "ceiling"}

# Pseudo semantic IDs.  These do not need to perfectly reproduce NYU40; they only
# make pseudo-ScanNet outputs usable by downstream code that expects integer labels.
SEMANTIC_ID = dict(getattr(geom, "PSEUDO_SEMANTIC_ID", {}))
SEMANTIC_ID.update({
    "wall": SEMANTIC_ID.get("wall", 1),
    "floor": SEMANTIC_ID.get("floor", 2),
    "cabinet_or_shelf": SEMANTIC_ID.get("cabinet", 3),
    "bed": SEMANTIC_ID.get("bed", 4),
    "chair": SEMANTIC_ID.get("chair", 5),
    "sofa": SEMANTIC_ID.get("sofa", 6),
    "table": SEMANTIC_ID.get("table", 7),
    "door": SEMANTIC_ID.get("door", 8),
    "window": SEMANTIC_ID.get("window", 9),
    "shelf_or_bookcase": SEMANTIC_ID.get("bookshelf", 10),
    "desk": SEMANTIC_ID.get("desk", 14),
    "work_desk": SEMANTIC_ID.get("desk", 14),
    "kitchen_counter": SEMANTIC_ID.get("counter", SEMANTIC_ID.get("cabinet", 3)),
    "stove": SEMANTIC_ID.get("otherfurniture", 39),
    "drawer": SEMANTIC_ID.get("dresser", SEMANTIC_ID.get("cabinet", 3)),
    "floor_lamp": SEMANTIC_ID.get("lamp", 35),
    "box": SEMANTIC_ID.get("otherprop", 40),
    "container": SEMANTIC_ID.get("otherprop", 40),
    "unknown_object": SEMANTIC_ID.get("unknown_object", 39),
    "unlabeled": SEMANTIC_ID.get("unlabeled", 0),
})

LABEL_ALIASES = {
    # sofa/couch
    "couch": "sofa",
    "loveseat": "sofa",
    "settee": "sofa",
    "sectional": "sofa",
    # beds
    "mattress": "bed",
    "futon": "bed",
    "bunk_bed": "bed",
    "crib": "bed",
    # desks/tables/counters
    "dining_table": "table",
    "coffee_table": "table",
    "side_table": "table",
    "nightstand": "table",
    "end_table": "table",
    "office_desk": "work_desk",
    "computer_desk": "work_desk",
    "workstation": "work_desk",
    "counter": "kitchen_counter",
    "countertop": "kitchen_counter",
    "kitchen_island": "kitchen_counter",
    "island": "kitchen_counter",
    # storage
    "cabinet": "cabinet_or_shelf",
    "shelf": "shelf_or_bookcase",
    "bookshelf": "shelf_or_bookcase",
    "bookcase": "shelf_or_bookcase",
    "dresser": "drawer",
    "drawers": "drawer",
    "chest_of_drawers": "drawer",
    # lights/containers
    "lamp": "floor_lamp",
    "standing_lamp": "floor_lamp",
    "floor_lamp_or_tall_thin_object": "floor_lamp",
    "storage_box": "box",
    "bin": "container",
    "basket": "container",
    "crate": "container",
    # architecture
    "walls": "wall",
    "windows": "window",
    "doors": "door",
}


def normalize_key(s: object) -> str:
    text = str(s or "").strip().lower()
    text = text.replace("-", "_").replace("/", "_").replace(" ", "_")
    while "__" in text:
        text = text.replace("__", "_")
    return text.strip("_")


def normalize_label(label: object) -> str:
    key = normalize_key(label)
    if not key:
        return "unknown_object"
    if key in TARGET_LABELS or key in {"floor", "ceiling"}:
        return key
    if key in LABEL_ALIASES:
        return LABEL_ALIASES[key]

    # Conservative substring matching for open-vocabulary model labels.
    if "window" in key:
        return "window"
    if "door" in key:
        return "door"
    if "wall" in key:
        return "wall"
    if "floor" in key:
        return "floor"
    if "ceiling" in key:
        return "ceiling"
    if "sofa" in key or "couch" in key or "sectional" in key:
        return "sofa"
    if "bed" in key or "mattress" in key:
        return "bed"
    if "stove" in key or "oven" in key or "range" in key or "cooktop" in key:
        return "stove"
    if "counter" in key or "island" in key:
        return "kitchen_counter"
    if "work" in key and "desk" in key:
        return "work_desk"
    if "desk" in key:
        return "desk"
    if "table" in key or "nightstand" in key:
        return "table"
    if "chair" in key or "stool" in key:
        return "chair"
    if "lamp" in key or "light" in key:
        return "floor_lamp"
    if "drawer" in key or "dresser" in key:
        return "drawer"
    if "book" in key and ("shelf" in key or "case" in key):
        return "shelf_or_bookcase"
    if "shelf" in key or "cabinet" in key or "closet" in key or "wardrobe" in key:
        return "cabinet_or_shelf"
    if "box" in key:
        return "box"
    if "container" in key or "bin" in key or "basket" in key or "crate" in key:
        return "container"
    return "unknown_object"


def angle_wrap(theta: float) -> float:
    while theta <= -math.pi:
        theta += 2.0 * math.pi
    while theta > math.pi:
        theta -= 2.0 * math.pi
    return float(theta)


def angle_diff_mod_pi(a: float, b: float) -> float:
    """Smallest absolute difference between orientations where theta == theta + pi."""
    return abs(angle_wrap(a - b + math.pi / 2.0) - math.pi / 2.0)


def bbox_corners(cx: float, cy: float, width: float, depth: float, theta: float) -> np.ndarray:
    hw, hd = width / 2.0, depth / 2.0
    local = np.array([[-hw, -hd], [hw, -hd], [hw, hd], [-hw, hd]], dtype=np.float64)
    c, s = math.cos(theta), math.sin(theta)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return local @ rot.T + np.array([cx, cy], dtype=np.float64)


def box_polygon(obj: Object2D) -> Polygon:
    pts = np.asarray(obj.footprint if obj.footprint else bbox_corners(obj.cx, obj.cy, obj.width, obj.depth, obj.theta))
    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def robust_oriented_bbox_2d(
    xy: np.ndarray,
    trim_percentile: float = 2.0,
    fixed_theta: Optional[float] = None,
) -> Tuple[np.ndarray, float, float, float, np.ndarray]:
    """Compute a robust 2D oriented box from mask points.

    PCA/min-rotated-rectangle boxes are often thrown off by a few stray scan
    points or small objects sitting on top of furniture.  This helper trims the
    extents in the box coordinate system.  For snapped orientations, pass
    fixed_theta to recompute the tight box around the same mask in that direction.
    """
    xy = np.asarray(xy, dtype=np.float64)
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) == 0:
        raise ValueError("Cannot box empty XY point set")
    if len(xy) < 3:
        mn, mx = xy.min(axis=0), xy.max(axis=0)
        center = (mn + mx) / 2.0
        width, depth = np.maximum(mx - mn, 0.05)
        theta = float(fixed_theta or 0.0)
        corners = bbox_corners(center[0], center[1], width, depth, theta)
        return center, float(width), float(depth), theta, corners

    mean = xy.mean(axis=0)
    if fixed_theta is None:
        centered = xy - mean
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        axis = eigvecs[:, int(np.argmax(eigvals))]
        theta = math.atan2(axis[1], axis[0])
    else:
        theta = float(fixed_theta)

    c, s = math.cos(-theta), math.sin(-theta)
    rot_to_box = np.array([[c, -s], [s, c]], dtype=np.float64)
    pts_box = (xy - mean) @ rot_to_box.T

    trim = max(0.0, min(float(trim_percentile), 20.0))
    if len(xy) >= 20 and trim > 0:
        lo = np.percentile(pts_box, trim, axis=0)
        hi = np.percentile(pts_box, 100.0 - trim, axis=0)
    else:
        lo = pts_box.min(axis=0)
        hi = pts_box.max(axis=0)

    # Guard against over-trimming thin sparse masks.
    raw_lo, raw_hi = pts_box.min(axis=0), pts_box.max(axis=0)
    if np.any((hi - lo) < 0.25 * np.maximum(raw_hi - raw_lo, 1e-6)):
        lo, hi = raw_lo, raw_hi

    width, depth = np.maximum(hi - lo, 0.05)
    center_box = (lo + hi) / 2.0
    c2, s2 = math.cos(theta), math.sin(theta)
    rot_to_world = np.array([[c2, -s2], [s2, c2]], dtype=np.float64)
    center = center_box @ rot_to_world.T + mean
    corners = bbox_corners(center[0], center[1], float(width), float(depth), theta)
    return center, float(width), float(depth), angle_wrap(theta), corners


def room_axes_from_polygon(room_poly: Polygon, min_edge_length: float = 0.50) -> List[float]:
    if room_poly.is_empty or not hasattr(room_poly, "exterior"):
        return [0.0, math.pi / 2.0]
    coords = np.asarray(room_poly.exterior.coords, dtype=np.float64)
    axes: List[Tuple[float, float]] = []
    for a, b in zip(coords[:-1], coords[1:]):
        v = b - a
        length = float(np.linalg.norm(v))
        if length < min_edge_length:
            continue
        theta = math.atan2(v[1], v[0])
        theta = angle_wrap(theta)
        # Keep theta in [0, pi) orientation space.
        if theta < 0:
            theta += math.pi
        if theta >= math.pi:
            theta -= math.pi
        axes.append((theta, length))

    if not axes:
        return [0.0, math.pi / 2.0]

    # Greedy weighted unique axes.  Room polygons are usually rectilinear, so two
    # perpendicular axes dominate.
    axes.sort(key=lambda t: t[1], reverse=True)
    selected: List[float] = []
    for theta, _length in axes:
        if all(angle_diff_mod_pi(theta, s) > math.radians(12.0) for s in selected):
            selected.append(theta)
        if len(selected) >= 4:
            break
    if not selected:
        selected = [0.0]
    # Always include perpendiculars because furniture may align width or depth to a wall.
    out: List[float] = []
    for theta in selected:
        for cand in (theta, theta + math.pi / 2.0):
            cand = angle_wrap(cand)
            if cand < 0:
                cand += math.pi
            if cand >= math.pi:
                cand -= math.pi
            if all(angle_diff_mod_pi(cand, x) > math.radians(5.0) for x in out):
                out.append(cand)
    return out or [0.0, math.pi / 2.0]


def nearest_room_axis(theta: float, axes: Sequence[float]) -> Tuple[float, float]:
    best_axis = 0.0
    best_diff = float("inf")
    for axis in axes:
        for cand in (axis, axis + math.pi / 2.0, axis - math.pi / 2.0):
            diff = angle_diff_mod_pi(theta, cand)
            if diff < best_diff:
                best_axis = angle_wrap(cand)
                best_diff = diff
    return best_axis, best_diff


def should_snap_label(label: str) -> bool:
    return label in {
        "bed", "sofa", "table", "desk", "work_desk", "kitchen_counter", "stove",
        "drawer", "cabinet_or_shelf", "shelf_or_bookcase", "door", "window",
    }


def parse_index_list(value: object, n_points: int) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.dtype == bool or (arr.ndim == 1 and arr.size == n_points and set(np.unique(arr).tolist()).issubset({0, 1, False, True})):
        return np.flatnonzero(arr.astype(bool)).astype(np.int64)
    arr = arr.astype(np.int64).reshape(-1)
    arr = arr[(arr >= 0) & (arr < n_points)]
    if len(arr) == 0:
        return np.empty((0,), dtype=np.int64)
    return np.unique(arr)


def get_instance_indices(inst: dict, n_points: int) -> Optional[np.ndarray]:
    for key in ("point_indices", "vertex_indices", "indices", "mask_indices"):
        if key in inst:
            return parse_index_list(inst.get(key), n_points)
    if "mask" in inst:
        return parse_index_list(inst.get("mask"), n_points)
    return None


def instance_label_and_score(inst: dict) -> Tuple[str, str, float]:
    raw_label = (
        inst.get("label") or inst.get("class_name") or inst.get("class") or
        inst.get("category") or inst.get("predicted_class") or "unknown_object"
    )
    label = normalize_label(raw_label)
    score = inst.get("score", inst.get("confidence", inst.get("probability", 1.0)))
    try:
        score_f = float(score)
    except Exception:
        score_f = 1.0
    return label, str(raw_label), score_f


def collect_semantic_masks(pred: dict, n_points: int) -> Dict[str, np.ndarray]:
    masks: Dict[str, List[np.ndarray]] = {}

    for key in ("semantic_regions", "semantic_masks", "regions"):
        for region in pred.get(key, []) or []:
            label = normalize_label(region.get("label") or region.get("class_name") or region.get("class"))
            idx = get_instance_indices(region, n_points)
            if idx is not None and len(idx) > 0:
                masks.setdefault(label, []).append(idx)

    # Also treat structural instances as semantic masks.
    for inst in pred.get("instances", []) or []:
        label, _raw, _score = instance_label_and_score(inst)
        if label in STRUCTURAL_LABELS:
            idx = get_instance_indices(inst, n_points)
            if idx is not None and len(idx) > 0:
                masks.setdefault(label, []).append(idx)

    out: Dict[str, np.ndarray] = {}
    for label, chunks in masks.items():
        if chunks:
            out[label] = np.unique(np.concatenate(chunks)).astype(np.int64)
    return out


def make_room_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        room_source=args.room_source,
        floor_height_threshold=args.floor_height_threshold,
        room_slice_max_z=args.room_slice_max_z,
        room_resolution_m=args.room_resolution_m,
        room_close_iters=args.room_close_iters,
        room_dilate_iters=args.room_dilate_iters,
        room_simplify_m=args.room_simplify_m,
        min_room_area=args.min_room_area,
    )


def estimate_aligned_points(args: argparse.Namespace) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray, Dict[str, object]]:
    raw_xyz, rgb = geom.load_user_ply_vertices(args.ply)
    rng = np.random.default_rng(args.seed)
    if args.assume_aligned:
        aligned = raw_xyz.astype(np.float32).copy()
        floor_z = float(np.percentile(aligned[:, 2], 1.0))
        aligned[:, 2] -= floor_z
        axis_alignment = np.eye(4, dtype=np.float32)
        axis_alignment[2, 3] = -floor_z
        meta = {"mode": "assume_aligned", "floor_z_translation": float(-floor_z)}
    else:
        aligned, axis_alignment, meta = geom.estimate_alignment(
            raw_xyz,
            up_axis=args.up_axis,
            plane_distance=args.plane_distance,
            ransac_iterations=args.ransac_iterations,
            rng=rng,
        )
    return raw_xyz, rgb, aligned.astype(np.float32), axis_alignment.astype(np.float32), meta


def room_polygon_from_model_or_geometry(
    aligned: np.ndarray,
    semantic_masks: Dict[str, np.ndarray],
    args: argparse.Namespace,
) -> Tuple[Polygon, Dict[str, object]]:
    if args.room_from == "model_floor" and "floor" in semantic_masks and len(semantic_masks["floor"]) >= 30:
        floor_xy = aligned[semantic_masks["floor"], :2]
        poly, debug = geom.raster_polygon_from_points(
            floor_xy,
            aligned[:, :2],
            resolution=args.room_resolution_m,
            close_iters=args.room_close_iters,
            dilate_iters=args.room_dilate_iters,
            simplify_m=args.room_simplify_m,
            min_area=args.min_room_area,
        )
        debug["source"] = "model_floor_mask"
        return poly, debug

    poly, debug = geom.estimate_room_polygon(aligned, make_room_args(args))
    debug["source"] = "geometry_fallback_lower_slice_or_floor"
    return poly, debug


def bbox_from_instance_mask(
    points: np.ndarray,
    label: str,
    room_axes: Sequence[float],
    args: argparse.Namespace,
) -> Tuple[np.ndarray, float, float, float, np.ndarray, float, float, float, Dict[str, object]]:
    xy = points[:, :2]
    center, width, depth, theta, corners = robust_oriented_bbox_2d(xy, trim_percentile=args.bbox_trim_percentile)
    snap_debug: Dict[str, object] = {"snapped": False, "initial_theta": float(theta)}

    if args.snap_orientations != "off" and should_snap_label(label):
        snapped_theta, diff = nearest_room_axis(theta, room_axes)
        diff_deg = math.degrees(diff)
        snap_debug.update({"nearest_axis": float(snapped_theta), "diff_deg": float(diff_deg)})
        if args.snap_orientations == "always" or diff_deg <= args.snap_angle_deg:
            center, width, depth, theta, corners = robust_oriented_bbox_2d(
                xy,
                trim_percentile=args.bbox_trim_percentile,
                fixed_theta=snapped_theta,
            )
            snap_debug.update({"snapped": True, "final_theta": float(theta)})

    z_trim = max(0.0, min(float(args.z_trim_percentile), 20.0))
    if len(points) >= 20 and z_trim > 0:
        z_min = float(np.percentile(points[:, 2], z_trim))
        z_max = float(np.percentile(points[:, 2], 100.0 - z_trim))
    else:
        z_min = float(points[:, 2].min())
        z_max = float(points[:, 2].max())
    height = max(z_max - z_min, 0.01)
    return center, float(width), float(depth), float(theta), corners, z_min, z_max, height, snap_debug


def bbox_from_prediction_bbox(inst: dict, raw_label: str, label: str, points: np.ndarray, args: argparse.Namespace) -> Tuple[Object2D, Dict[str, object]]:
    bbox = inst.get("bbox") or inst.get("box") or inst.get("bbox2d") or {}
    cx = float(bbox.get("cx", bbox.get("x", 0.0)))
    cy = float(bbox.get("cy", bbox.get("y", 0.0)))
    width = float(bbox.get("width", bbox.get("w", 0.5)))
    depth = float(bbox.get("depth", bbox.get("d", 0.5)))
    theta = float(bbox.get("theta", bbox.get("yaw", 0.0)))
    z_min = float(bbox.get("z_min", bbox.get("zmin", 0.0)))
    z_max = float(bbox.get("z_max", bbox.get("zmax", bbox.get("height", 0.5))))
    if "height" in bbox and "z_max" not in bbox and "zmax" not in bbox:
        z_max = z_min + float(bbox["height"])
    height = max(z_max - z_min, 0.01)
    corners = bbox_corners(cx, cy, width, depth, theta)
    center_3d = [float(cx), float(cy), float((z_min + z_max) / 2.0)]
    size_3d = [float(width), float(depth), float(height)]
    obj = Object2D(
        id=0,
        object_id=0,
        label=label,
        raw_label=str(raw_label),
        cx=cx,
        cy=cy,
        width=max(width, 0.05),
        depth=max(depth, 0.05),
        theta=angle_wrap(theta),
        z_min=z_min,
        z_max=z_max,
        height=height,
        point_count=0,
        footprint=[[float(x), float(y)] for x, y in corners.tolist()],
        bbox3d_center=center_3d,
        bbox3d_size=size_3d,
    )
    return obj, {"bbox_source": "model_bbox_no_mask"}


def outside_ratio(obj: Object2D, room_poly: Polygon, tolerance: float) -> float:
    if room_poly.is_empty:
        return 0.0
    poly = box_polygon(obj)
    if poly.is_empty or poly.area <= 1e-9:
        return 0.0
    outside = poly.difference(room_poly.buffer(tolerance)).area
    return float(outside / max(poly.area, 1e-9))


def nudge_object_toward_room(obj: Object2D, room_poly: Polygon, max_ratio: float, tol: float) -> bool:
    if room_poly.is_empty:
        return True
    start = np.array([obj.cx, obj.cy], dtype=np.float64)
    target_point = room_poly.representative_point()
    target = np.array([target_point.x, target_point.y], dtype=np.float64)
    vec = target - start
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return outside_ratio(obj, room_poly, tol) <= max_ratio
    vec /= norm

    original = (obj.cx, obj.cy, list(obj.footprint))
    max_step = max(room_poly.bounds[2] - room_poly.bounds[0], room_poly.bounds[3] - room_poly.bounds[1], 1.0)
    best = (outside_ratio(obj, room_poly, tol), obj.cx, obj.cy)
    for step in np.linspace(0.0, max_step, 80):
        obj.cx = float(start[0] + vec[0] * step)
        obj.cy = float(start[1] + vec[1] * step)
        obj.footprint = [[float(x), float(y)] for x, y in bbox_corners(obj.cx, obj.cy, obj.width, obj.depth, obj.theta).tolist()]
        ratio = outside_ratio(obj, room_poly, tol)
        if ratio < best[0]:
            best = (ratio, obj.cx, obj.cy)
        if ratio <= max_ratio and room_poly.buffer(tol).contains(Point(obj.cx, obj.cy)):
            obj.bbox3d_center = [float(obj.cx), float(obj.cy), float((obj.z_min + obj.z_max) / 2.0)]
            return True
    obj.cx, obj.cy = best[1], best[2]
    obj.footprint = [[float(x), float(y)] for x, y in bbox_corners(obj.cx, obj.cy, obj.width, obj.depth, obj.theta).tolist()]
    obj.bbox3d_center = [float(obj.cx), float(obj.cy), float((obj.z_min + obj.z_max) / 2.0)]
    return best[0] <= max_ratio


def build_objects_from_predictions(
    pred: dict,
    aligned: np.ndarray,
    room_poly: Polygon,
    args: argparse.Namespace,
) -> Tuple[List[Object2D], Dict[str, object], np.ndarray, np.ndarray, List[dict]]:
    n = len(aligned)
    room_axes = room_axes_from_polygon(room_poly, min_edge_length=args.min_wall_axis_length)
    objects: List[Object2D] = []
    debug_instances: List[dict] = []

    seg_indices = np.zeros(n, dtype=np.int32)
    semantic_labels = np.full(n, SEMANTIC_ID.get("unlabeled", 0), dtype=np.int32)
    seg_groups: List[dict] = []

    semantic_masks = collect_semantic_masks(pred, n)

    # If the model did not explicitly return floor/wall/ceiling masks, create a
    # conservative geometry fallback for the pseudo-ScanNet labels.  This does
    # not create furniture objects; it only keeps downstream ScanNet-style files
    # from being entirely unlabeled outside the predicted instances.
    if "floor" not in semantic_masks:
        semantic_masks["floor"] = np.flatnonzero(aligned[:, 2] <= args.floor_height_threshold).astype(np.int64)

    for label, idx in semantic_masks.items():
        semantic_labels[idx] = SEMANTIC_ID.get(label, SEMANTIC_ID.get("unknown_object", 39))
        if label == "floor":
            seg_indices[idx] = 0
        elif label == "wall":
            seg_indices[idx] = 1
        elif label == "ceiling":
            seg_indices[idx] = 2

    next_id = 1
    next_seg = 10
    for inst_i, inst in enumerate(pred.get("instances", []) or [], start=1):
        label, raw_label, score = instance_label_and_score(inst)
        idx = get_instance_indices(inst, n)

        if label in NON_OBJECT_STRUCTURAL and not args.include_structural:
            debug_instances.append({"instance": inst_i, "raw_label": raw_label, "label": label, "status": "skipped_non_object_structural"})
            continue
        if label == "wall" and not args.include_walls_as_objects:
            # Keep wall semantic points in pseudo labels, but don't put every wall in the furniture editor.
            debug_instances.append({"instance": inst_i, "raw_label": raw_label, "label": label, "status": "skipped_wall_object"})
            continue

        if idx is not None and len(idx) > 0:
            if len(idx) < args.min_model_points and label not in ARCHITECTURE_LABELS:
                debug_instances.append({"instance": inst_i, "raw_label": raw_label, "label": label, "points": int(len(idx)), "status": "skipped_too_few_points"})
                continue
            pts = aligned[idx]
            center, width, depth, theta, corners, z_min, z_max, height, snap_debug = bbox_from_instance_mask(
                pts, label, room_axes, args
            )
            center_3d = pts.mean(axis=0)
            size_3d = pts.max(axis=0) - pts.min(axis=0)
            point_count = int(len(idx))
            source = "model_mask"
        elif any(k in inst for k in ("bbox", "box", "bbox2d")):
            obj, snap_debug = bbox_from_prediction_bbox(inst, raw_label, label, aligned, args)
            center = np.array([obj.cx, obj.cy], dtype=np.float64)
            width, depth, theta = obj.width, obj.depth, obj.theta
            corners = np.asarray(obj.footprint, dtype=np.float64)
            z_min, z_max, height = obj.z_min, obj.z_max, obj.height
            center_3d = np.asarray(obj.bbox3d_center, dtype=np.float64)
            size_3d = np.asarray(obj.bbox3d_size, dtype=np.float64)
            point_count = 0
            idx = np.empty((0,), dtype=np.int64)
            source = "model_bbox"
        else:
            debug_instances.append({"instance": inst_i, "raw_label": raw_label, "label": label, "status": "skipped_no_mask_or_bbox"})
            continue

        obj = Object2D(
            id=next_id,
            object_id=int(inst.get("id", inst.get("object_id", next_id))) if str(inst.get("id", inst.get("object_id", next_id))).lstrip("-").isdigit() else next_id,
            label=label,
            raw_label=str(raw_label),
            cx=float(center[0]),
            cy=float(center[1]),
            width=float(max(width + args.box_padding * 2.0, 0.05)),
            depth=float(max(depth + args.box_padding * 2.0, 0.05)),
            theta=float(theta),
            z_min=float(z_min),
            z_max=float(z_max),
            height=float(height),
            point_count=point_count,
            footprint=[[float(x), float(y)] for x, y in bbox_corners(float(center[0]), float(center[1]), float(max(width + args.box_padding * 2.0, 0.05)), float(max(depth + args.box_padding * 2.0, 0.05)), float(theta)).tolist()],
            bbox3d_center=[float(v) for v in np.asarray(center_3d).tolist()],
            bbox3d_size=[float(v) for v in np.asarray(size_3d).tolist()],
        )

        ratio = outside_ratio(obj, room_poly, args.outside_room_tolerance)
        status = "kept"
        if label not in ARCHITECTURE_LABELS and ratio > args.max_outside_ratio:
            if args.outside_policy == "reject":
                debug_instances.append({
                    "instance": inst_i, "raw_label": raw_label, "label": label, "status": "rejected_outside_room",
                    "outside_ratio": float(ratio), "source": source,
                })
                continue
            if args.outside_policy == "nudge":
                ok = nudge_object_toward_room(obj, room_poly, args.max_outside_ratio, args.outside_room_tolerance)
                status = "nudged_inside" if ok else "kept_after_failed_nudge"

        if idx is not None and len(idx) > 0:
            seg_indices[idx] = next_seg
            semantic_labels[idx] = SEMANTIC_ID.get(label, SEMANTIC_ID.get("unknown_object", 39))
        seg_groups.append({"objectId": int(next_id), "id": int(next_id), "label": label, "segments": [int(next_seg)]})
        objects.append(obj)
        debug_instances.append({
            "instance": inst_i,
            "raw_label": raw_label,
            "label": label,
            "score": float(score),
            "points": point_count,
            "source": source,
            "status": status,
            "outside_ratio": float(outside_ratio(obj, room_poly, args.outside_room_tolerance)),
            "theta": float(obj.theta),
            "snap": snap_debug,
            "width": float(obj.width),
            "depth": float(obj.depth),
        })
        next_id += 1
        next_seg += 1

    objects.sort(key=lambda o: (0 if o.label in ARCHITECTURE_LABELS else 1, -(o.width * o.depth), o.label, o.id))
    for new_id, obj in enumerate(objects, start=1):
        obj.id = new_id

    pseudo = {
        "seg_indices": seg_indices,
        "semantic_labels": semantic_labels,
        "seg_groups": seg_groups,
        "floor_mask": semantic_masks.get("floor", np.flatnonzero(aligned[:, 2] <= args.floor_height_threshold)),
        "wall_mask": semantic_masks.get("wall", np.zeros((0,), dtype=np.int64)),
        "ceiling_mask": semantic_masks.get("ceiling", np.zeros((0,), dtype=np.int64)),
        "candidate_mask": np.flatnonzero(seg_indices >= 10),
        "wall_planes": [],
    }
    debug = {
        "room_axes_radians": [float(x) for x in room_axes],
        "room_axes_degrees": [float(math.degrees(x)) for x in room_axes],
        "instances": debug_instances,
        "object_count": int(len(objects)),
    }
    return objects, pseudo, semantic_labels, seg_indices, seg_groups, debug


def run_inference_if_requested(args: argparse.Namespace) -> Optional[Path]:
    if not args.inference_cmd:
        return None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = Path(args.predictions_json) if args.predictions_json else out_dir / "model_predictions.json"
    cmd = args.inference_cmd.format(
        ply=str(Path(args.ply)),
        out_json=str(pred_path),
        predictions=str(pred_path),
        out_dir=str(out_dir),
        scene_id=args.scene_id or Path(args.ply).stem,
    )
    print(f"[model-topdown] running inference command:\n  {cmd}")
    subprocess.run(cmd, shell=True, check=True)
    if not pred_path.exists():
        raise FileNotFoundError(f"Inference command completed, but predictions JSON was not created: {pred_path}")
    return pred_path


def read_predictions(args: argparse.Namespace) -> Tuple[dict, Path]:
    pred_from_cmd = run_inference_if_requested(args)
    pred_path = pred_from_cmd or (Path(args.predictions_json) if args.predictions_json else None)
    if pred_path is None:
        raise ValueError(
            "No model predictions were provided. Use --predictions_json, or use --inference_cmd to run an external model. "
            "The geometry-only fallback remains in export_user_ply_topdown_v5.py."
        )
    with Path(pred_path).open("r", encoding="utf-8") as f:
        pred = json.load(f)
    if "instances" not in pred:
        raise ValueError(f"Predictions JSON must contain an 'instances' list: {pred_path}")
    return pred, Path(pred_path)


def build_layout(args: argparse.Namespace) -> Tuple[SceneLayout, Dict[str, object], np.ndarray, Optional[np.ndarray], Dict[str, object]]:
    pred, pred_path = read_predictions(args)
    raw_xyz, rgb, aligned, axis_alignment, align_meta = estimate_aligned_points(args)

    semantic_masks = collect_semantic_masks(pred, len(aligned))
    room_poly, room_debug = room_polygon_from_model_or_geometry(aligned, semantic_masks, args)
    objects, pseudo, _semantic_labels, _seg_indices, _seg_groups, object_debug = build_objects_from_predictions(
        pred, aligned, room_poly, args
    )
    pseudo["axis_alignment"] = axis_alignment

    scene_id = args.scene_id or Path(args.ply).stem.replace(" ", "_")
    layout = SceneLayout(
        scene_id=scene_id,
        units="meters",
        room_polygon=polygon_to_coords(room_poly),
        room_bbox=polygon_bbox(room_poly),
        objects=objects,
        metadata={
            "source": "user_ply_model_backed_topdown",
            "input_ply": str(Path(args.ply)),
            "predictions_json": str(pred_path),
            "axis_alignment_estimated": not args.assume_aligned,
            "axis_alignment_matrix": [float(x) for x in axis_alignment.reshape(-1).tolist()],
            "alignment": align_meta,
            "coordinate_frame_declared_by_predictions": pred.get("coordinate_frame", "input_or_indices"),
            "room_from": args.room_from,
            "room_polygon_method": room_debug,
            "object_count": len(objects),
            "label_set": TARGET_LABELS,
            "bbox_method": "model_mask_robust_trimmed_obb_with_optional_wall_axis_snap",
            "snap_orientations": args.snap_orientations,
            "outside_policy": args.outside_policy,
            "important_note": (
                "This exporter assumes semantic/instance recognition comes from model predictions. "
                "It intentionally avoids geometry-only relabeling except for normalization of model labels."
            ),
        },
    )
    debug = {"room_debug": room_debug, "object_debug": object_debug}
    return layout, pseudo, aligned, rgb, debug


def write_debug_outputs(out_dir: Path, layout: SceneLayout, aligned: np.ndarray, pseudo: dict, debug: dict, args: argparse.Namespace) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "debug_model_topdown_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"metadata": layout.metadata, **debug}, f, indent=2)

    # Overlay final boxes over all aligned source points to verify coordinate preservation.
    rng = np.random.default_rng(args.seed)
    n = len(aligned)
    sample_n = min(n, args.debug_point_sample)
    idx = rng.choice(n, size=sample_n, replace=False) if n > sample_n else np.arange(n)
    pts = aligned[idx]

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.scatter(pts[:, 0], pts[:, 1], s=0.25, alpha=0.20, label="aligned .ply points")
    room = np.asarray(layout.room_polygon, dtype=float)
    if len(room) >= 3:
        closed = np.vstack([room, room[0]])
        ax.plot(closed[:, 0], closed[:, 1], linewidth=2.0, label="room polygon")
    for obj in layout.objects:
        corners = np.asarray(obj.footprint, dtype=float)
        if len(corners) >= 3:
            closed = np.vstack([corners, corners[0]])
            ax.plot(closed[:, 0], closed[:, 1], linewidth=1.4)
            ax.text(obj.cx, obj.cy, obj.label, fontsize=7, ha="center", va="center")
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Model top-down debug: final boxes over aligned source .ply points")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "debug_model_points_overlay.png", dpi=200)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert model predictions on a user .ply scan into the shared top-down layout format.")
    p.add_argument("--ply", required=True, help="Raw user .ply room scan. This remains the only user-provided room input.")
    p.add_argument("--predictions_json", help="Model predictions JSON in the schema documented at the top of this file.")
    p.add_argument("--inference_cmd", help="Optional external command that writes predictions JSON. Use {ply}, {predictions}, {out_dir}, {scene_id} placeholders.")
    p.add_argument("--out_dir", required=True, help="Output directory.")
    p.add_argument("--scene_id", help="Scene id/name for output files.")

    # Alignment / preprocessing
    p.add_argument("--assume_aligned", action="store_true", help="Assume the .ply is already z-up; only translate floor near z=0.")
    p.add_argument("--up_axis", default="z", choices=["x", "y", "z"], help="Fallback up axis before floor-plane alignment.")
    p.add_argument("--plane_distance", type=float, default=0.035, help="RANSAC plane inlier threshold for floor alignment.")
    p.add_argument("--ransac_iterations", type=int, default=1500)
    p.add_argument("--seed", type=int, default=7)

    # Room polygon
    p.add_argument("--room_from", default="auto", choices=["auto", "model_floor"], help="Use model floor mask for room polygon if available, otherwise use geometry fallback.")
    p.add_argument("--room_source", default="auto", choices=["auto", "floor", "lower_slice"])
    p.add_argument("--floor_height_threshold", type=float, default=0.06)
    p.add_argument("--room_slice_max_z", type=float, default=0.35)
    p.add_argument("--room_resolution_m", type=float, default=0.04)
    p.add_argument("--room_close_iters", type=int, default=0, help="Keep low to preserve concave/inset corners.")
    p.add_argument("--room_dilate_iters", type=int, default=0)
    p.add_argument("--room_simplify_m", type=float, default=0.005, help="Keep low to avoid smoothing inset corners.")
    p.add_argument("--min_room_area", type=float, default=0.50)

    # Object conversion
    p.add_argument("--min_model_points", type=int, default=30, help="Skip non-architecture model masks smaller than this.")
    p.add_argument("--bbox_trim_percentile", type=float, default=2.0, help="Trim this percent of XY outliers on each side while boxing masks.")
    p.add_argument("--z_trim_percentile", type=float, default=1.0, help="Trim vertical outliers to ignore small clutter on top of objects.")
    p.add_argument("--box_padding", type=float, default=0.03, help="Add small padding around mask-derived boxes, in meters.")
    p.add_argument("--include_structural", action="store_true", help="Include floor/ceiling as editor objects. Usually false.")
    p.add_argument("--include_walls_as_objects", action="store_true", help="Include wall instances as editor objects. Usually false; wall polygon is already shown separately.")

    # Orientation and outside handling
    p.add_argument("--snap_orientations", default="near", choices=["off", "near", "always"], help="Snap boxes to dominant wall axes.")
    p.add_argument("--snap_angle_deg", type=float, default=25.0, help="Maximum angle difference for --snap_orientations near.")
    p.add_argument("--min_wall_axis_length", type=float, default=0.50)
    p.add_argument("--outside_policy", default="reject", choices=["keep", "reject", "nudge"], help="What to do with non-architecture boxes mostly outside the room polygon.")
    p.add_argument("--outside_room_tolerance", type=float, default=0.12)
    p.add_argument("--max_outside_ratio", type=float, default=0.45)

    # Output
    p.add_argument("--no_png", action="store_true")
    p.add_argument("--no_debug", action="store_true")
    p.add_argument("--no_pseudo_scannet", action="store_true")
    p.add_argument("--no_copy_original", action="store_true")
    p.add_argument("--debug_point_sample", type=int, default=80000)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layout, pseudo, aligned, rgb, debug = build_layout(args)
    save_layout_json(layout, out_dir / "scene_layout.json")
    if not args.no_png:
        render_layout_png(layout, out_dir / "scene_layout.png")
    if not args.no_pseudo_scannet:
        geom.write_pseudo_scannet_files(
            out_dir,
            layout.scene_id,
            args.ply,
            aligned,
            rgb,
            pseudo,
            copy_original=not args.no_copy_original,
        )
    if not args.no_debug:
        write_debug_outputs(out_dir, layout, aligned, pseudo, debug, args)

    print(f"Wrote {out_dir / 'scene_layout.json'}")
    if not args.no_png:
        print(f"Wrote {out_dir / 'scene_layout.png'}")
    if not args.no_debug:
        print(f"Wrote {out_dir / 'debug_model_topdown_summary.json'}")
        print(f"Wrote {out_dir / 'debug_model_points_overlay.png'}")


if __name__ == "__main__":
    main()
