#!/usr/bin/env python3
"""
Shared user .ply geometry helpers for pseudo-ScanNet + top-down draggable room layout export.

Why this module exists
-------------
v1/v2 intentionally used geometry-only object proposals, but real phone/LiDAR
PLY scans are sparse, noisy, and semantically unlabeled. A single connected
occupancy component can span half the room, while true furniture objects often
show up only as partial surfaces. This version is designed to be higher-recall
and easier to debug:

  * room polygon is estimated from a raster footprint, not a convex hull, so
    inset/concave corners are preserved much better;
  * object proposals are generated in multiple height bands, then merged with
    non-maximum suppression, instead of relying on one global component pass;
  * giant room-sized components are rejected instead of exported as furniture;
  * labels are more detailed geometry-based guesses, with the expectation that
    these are later replaced by a ScanNet-trained semantic/instance model.

The output contract matches the earlier ScanNet exporter:

  out_dir/scene_layout.json
  out_dir/scene_layout.png
  out_dir/pseudo_scannet/scans/<scene_id>/<scene_id>_vh_clean_2.ply
  out_dir/pseudo_scannet/scans/<scene_id>/<scene_id>.aggregation.json
  out_dir/pseudo_scannet/scans/<scene_id>/<scene_id>_vh_clean_2.0.010000.segs.json
  out_dir/pseudo_scannet/scans/<scene_id>/<scene_id>_vh_clean_2.labels.ply
  out_dir/pseudo_scannet/scans/<scene_id>/<scene_id>.txt
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
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes, label as cc_label
from shapely.geometry import MultiPoint, Polygon, box
from shapely.ops import unary_union

from export_scannet_topdown import (
    Object2D,
    apply_transform,
    SceneLayout,
    oriented_bbox_2d,
    polygon_bbox,
    polygon_to_coords,
    read_vertex_table_from_ply,
    render_layout_png,
    save_layout_json,
)
SEMANTIC_ID = {
    "wall": 1,
    "floor": 2,
    "cabinet": 3,
    "bed": 4,
    "chair": 5,
    "sofa": 6,
    "table": 7,
    "door": 8,
    "window": 9,
    "bookshelf": 10,
    "desk": 14,
    "shelf_or_cabinet": 3,
    "bed_or_sofa": 6,
    "table_or_desk": 14,
    "chair_or_small_furniture": 5,
    "large_furniture": 39,
    "small_furniture": 39,
    "unknown_object": 39,
    "ceiling": 22,
    "unlabeled": 0,
}


def write_xyzrgb_ply(path: str | Path, xyz: np.ndarray, rgb: Optional[np.ndarray] = None) -> None:
    """Write an ASCII PLY with x/y/z and optional RGB. Faces are intentionally omitted."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(xyz)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if rgb is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        if rgb is None:
            for p in xyz:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        else:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)
            for p, c in zip(xyz, rgb):
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def write_labels_ply(path: str | Path, labels: np.ndarray) -> None:
    """Write a minimal ScanNet-like labels PLY with one label per vertex."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = labels.astype(np.int32)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(labels)}\n")
        f.write("property int label\n")
        f.write("end_header\n")
        for label in labels:
            f.write(f"{int(label)}\n")


def normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v
    return v / n


def rotation_matrix_from_vectors(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Return 3x3 rotation matrix that rotates src vector onto dst vector."""
    a = normalize(np.asarray(src, dtype=np.float64))
    b = normalize(np.asarray(dst, dtype=np.float64))
    v = np.cross(a, b)
    c = float(np.dot(a, b))

    if c > 1.0 - 1e-8:
        return np.eye(3, dtype=np.float64)

    if c < -1.0 + 1e-8:
        # Pick any axis perpendicular to a.
        axis = np.array([1.0, 0.0, 0.0])
        if abs(a[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0])
        v = normalize(np.cross(a, axis))
        vx = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0],
        ])
        return np.eye(3) + 2 * (vx @ vx)

    s = float(np.linalg.norm(v))
    vx = np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0],
    ])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s ** 2))


def make_transform(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.astype(np.float32)
    T[:3, 3] = t.astype(np.float32)
    return T


def sample_points(points: np.ndarray, max_points: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    if len(points) <= max_points:
        idx = np.arange(len(points), dtype=np.int64)
        return points, idx
    idx = rng.choice(len(points), size=max_points, replace=False)
    return points[idx], idx.astype(np.int64)


def voxel_downsample_indices(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Return representative original indices for occupied voxels."""
    if voxel_size <= 0:
        return np.arange(len(points), dtype=np.int64)
    mins = points.min(axis=0)
    vox = np.floor((points - mins) / voxel_size).astype(np.int64)
    _, first = np.unique(vox, axis=0, return_index=True)
    return np.sort(first.astype(np.int64))


def fit_plane_from_three(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Optional[Tuple[np.ndarray, float]]:
    n = np.cross(p2 - p1, p3 - p1)
    norm = np.linalg.norm(n)
    if norm < 1e-8:
        return None
    n = n / norm
    d = -float(np.dot(n, p1))
    return n.astype(np.float64), d


def ransac_plane(
    points: np.ndarray,
    distance_threshold: float,
    iterations: int,
    rng: np.random.Generator,
    prefer_normal: Optional[np.ndarray] = None,
    prefer_weight: float = 0.0,
) -> Tuple[np.ndarray, float, np.ndarray]:
    """Fit a plane with RANSAC. Returns normal, d, and boolean inlier mask."""
    if len(points) < 3:
        raise ValueError("Need at least 3 points for plane fitting")

    best_score = -1.0
    best_normal = np.array([0.0, 0.0, 1.0])
    best_d = 0.0
    best_mask = np.zeros(len(points), dtype=bool)

    for _ in range(iterations):
        ids = rng.choice(len(points), size=3, replace=False)
        fit = fit_plane_from_three(points[ids[0]], points[ids[1]], points[ids[2]])
        if fit is None:
            continue
        normal, d = fit
        dist = np.abs(points @ normal + d)
        mask = dist <= distance_threshold
        count = int(mask.sum())
        if count < 3:
            continue
        score = float(count)
        if prefer_normal is not None and prefer_weight > 0:
            score += prefer_weight * count * abs(float(np.dot(normal, normalize(prefer_normal))))
        if score > best_score:
            best_score = score
            best_normal = normal
            best_d = d
            best_mask = mask

    if best_score < 0:
        raise RuntimeError("RANSAC failed to find a plane")

    # Refine plane by PCA over inliers.
    inlier_pts = points[best_mask]
    centroid = inlier_pts.mean(axis=0)
    centered = inlier_pts - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = normalize(vh[-1])
    d = -float(np.dot(normal, centroid))
    dist = np.abs(points @ normal + d)
    mask = dist <= distance_threshold
    return normal.astype(np.float64), d, mask


def axis_vector(axis: str) -> Optional[np.ndarray]:
    axis = axis.lower()
    if axis == "x":
        return np.array([1.0, 0.0, 0.0])
    if axis == "y":
        return np.array([0.0, 1.0, 0.0])
    if axis == "z":
        return np.array([0.0, 0.0, 1.0])
    if axis == "auto":
        return None
    raise ValueError("--up_axis must be one of auto, x, y, z")


def estimate_alignment(
    points: np.ndarray,
    up_axis: str,
    plane_distance: float,
    ransac_iterations: int,
    rng: np.random.Generator,
    max_ransac_points: int = 60000,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """Estimate transform from raw coordinates into z-up, floor-at-z=0 coordinates."""
    sampled, sampled_idx = sample_points(points, max_ransac_points, rng)
    preferred = axis_vector(up_axis)

    if preferred is None:
        normal, d, mask = ransac_plane(
            sampled,
            distance_threshold=plane_distance,
            iterations=ransac_iterations,
            rng=rng,
        )
    else:
        normal, d, mask = ransac_plane(
            sampled,
            distance_threshold=plane_distance,
            iterations=ransac_iterations,
            rng=rng,
            prefer_normal=preferred,
            prefer_weight=0.5,
        )
        if abs(float(np.dot(normal, preferred))) < 0.45:
            print(
                "WARNING: detected floor plane normal is not close to the requested up axis. "
                "Try --up_axis auto, --up_axis z, or --up_axis y depending on your scanner export."
            )

    # Orient normal toward the preferred up axis when known, otherwise pick the
    # sign that places most points above the plane after alignment.
    if preferred is not None and np.dot(normal, preferred) < 0:
        normal = -normal
        d = -d

    R = rotation_matrix_from_vectors(normal, np.array([0.0, 0.0, 1.0]))
    rotated = points @ R.T
    sampled_rotated = sampled @ R.T

    # Compute floor height from original plane in rotated space. Use inliers to
    # be robust; after alignment, inlier z values should be nearly constant.
    floor_z = float(np.median(sampled_rotated[mask, 2])) if int(mask.sum()) else float(np.percentile(rotated[:, 2], 1))
    t = np.array([0.0, 0.0, -floor_z], dtype=np.float64)
    T = make_transform(R, t)
    aligned = apply_transform(points.astype(np.float32), T)

    # If most points are below floor because normal sign was wrong, flip once.
    if np.percentile(aligned[:, 2], 90) < 0:
        R = rotation_matrix_from_vectors(-normal, np.array([0.0, 0.0, 1.0]))
        rotated = points @ R.T
        floor_z = float(np.median((sampled @ R.T)[mask, 2])) if int(mask.sum()) else float(np.percentile(rotated[:, 2], 1))
        t = np.array([0.0, 0.0, -floor_z], dtype=np.float64)
        T = make_transform(R, t)
        aligned = apply_transform(points.astype(np.float32), T)
        normal = -normal
        d = -d

    meta = {
        "raw_floor_plane_normal": [float(x) for x in normal.tolist()],
        "raw_floor_plane_d": float(d),
        "floor_inliers_sample": int(mask.sum()),
        "floor_inlier_ratio_sample": float(mask.mean()),
        "up_axis_requested": up_axis,
        "floor_z_translation_after_rotation": float(-floor_z),
    }
    return aligned.astype(np.float32), T, meta


def detect_wall_planes(
    aligned_points: np.ndarray,
    base_mask: np.ndarray,
    plane_distance: float,
    ransac_iterations: int,
    rng: np.random.Generator,
    max_planes: int = 8,
    max_points: int = 50000,
    min_inlier_ratio: float = 0.015,
    min_height: float = 1.0,
    min_span: float = 1.0,
) -> Tuple[List[Tuple[np.ndarray, float]], np.ndarray]:
    """Find large vertical wall-like planes and return raw-size wall mask."""
    candidate_idx = np.where(base_mask)[0]
    if len(candidate_idx) < 100:
        return [], np.zeros(len(aligned_points), dtype=bool)

    sampled_idx = candidate_idx
    if len(sampled_idx) > max_points:
        sampled_idx = rng.choice(sampled_idx, size=max_points, replace=False)
    sampled = aligned_points[sampled_idx]
    remaining_mask = np.ones(len(sampled), dtype=bool)

    planes: List[Tuple[np.ndarray, float]] = []
    for _ in range(max_planes):
        pts = sampled[remaining_mask]
        if len(pts) < 500:
            break
        try:
            normal, d_local, inlier_local = ransac_plane(
                pts,
                distance_threshold=plane_distance,
                iterations=max(80, ransac_iterations // 4),
                rng=rng,
            )
        except Exception:
            break

        # Convert d from local point coordinates? Points are same global coords,
        # so d_local is already global.
        if abs(float(normal[2])) > 0.35:
            # horizontal-ish: not a wall
            local_indices = np.where(remaining_mask)[0]
            if int(inlier_local.sum()) > 0:
                remaining_mask[local_indices[inlier_local]] = False
            continue

        inlier_pts = pts[inlier_local]
        if len(inlier_pts) < max(200, int(min_inlier_ratio * len(sampled))):
            break

        z_span = float(inlier_pts[:, 2].max() - inlier_pts[:, 2].min())
        xy_span = float(np.linalg.norm(inlier_pts[:, :2].max(axis=0) - inlier_pts[:, :2].min(axis=0)))
        if z_span >= min_height and xy_span >= min_span:
            planes.append((normal, float(d_local)))

        local_indices = np.where(remaining_mask)[0]
        remaining_mask[local_indices[inlier_local]] = False

    wall_mask = np.zeros(len(aligned_points), dtype=bool)
    for normal, d in planes:
        dist = np.abs(aligned_points @ normal + d)
        wall_mask |= dist <= plane_distance

    return planes, wall_mask




# Additional label-to-NYU-ish fallback mapping used only for pseudo labels.
# ScanNet's real label map is not being reproduced here; these are just helpful
# enough for downstream code that expects integer semantic labels.
PSEUDO_SEMANTIC_ID = dict(SEMANTIC_ID)
PSEUDO_SEMANTIC_ID.update({
    "bed": SEMANTIC_ID.get("bed", 4),
    "sofa": SEMANTIC_ID.get("sofa", 6),
    "table": SEMANTIC_ID.get("table", 7),
    "desk": SEMANTIC_ID.get("desk", 14),
    "work_desk": SEMANTIC_ID.get("desk", 14),
    "kitchen_counter": SEMANTIC_ID.get("cabinet", 3),
    "counter_or_cabinet": SEMANTIC_ID.get("cabinet", 3),
    "cabinet_or_shelf": SEMANTIC_ID.get("cabinet", 3),
    "shelf_or_bookcase": SEMANTIC_ID.get("bookshelf", 10),
    "floor_lamp_or_tall_thin_object": SEMANTIC_ID.get("unknown_object", 39),
    "chair_or_stool": SEMANTIC_ID.get("chair", 5),
    "furniture_proposal": SEMANTIC_ID.get("unknown_object", 39),
})


class Proposal:
    def __init__(
        self,
        vertex_ids: np.ndarray,
        band_name: str,
        center: np.ndarray,
        width: float,
        depth: float,
        theta: float,
        corners: np.ndarray,
        z_min: float,
        z_max: float,
        height: float,
        label: str,
        score: float,
        reject_reason: Optional[str] = None,
    ) -> None:
        self.vertex_ids = vertex_ids.astype(np.int64)
        self.band_name = band_name
        self.center = center.astype(float)
        self.width = float(width)
        self.depth = float(depth)
        self.theta = float(theta)
        self.corners = corners.astype(float)
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.height = float(height)
        self.label = label
        self.score = float(score)
        self.reject_reason = reject_reason

    @property
    def area(self) -> float:
        return float(max(self.width, 0.0) * max(self.depth, 0.0))

    @property
    def poly(self) -> Polygon:
        p = Polygon(self.corners)
        if not p.is_valid:
            p = p.buffer(0)
        return p


def load_user_ply_vertices(path: str | Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    props = read_vertex_table_from_ply(path, None)
    for key in ("x", "y", "z"):
        if key not in props:
            raise ValueError(f"PLY file is missing vertex property '{key}': {path}")
    xyz = np.column_stack([props["x"], props["y"], props["z"]]).astype(np.float32)
    rgb = None
    if all(k in props for k in ("red", "green", "blue")):
        rgb = np.column_stack([props["red"], props["green"], props["blue"]]).astype(np.uint8)
    elif all(k in props for k in ("r", "g", "b")):
        rgb = np.column_stack([props["r"], props["g"], props["b"]]).astype(np.uint8)
    finite = np.isfinite(xyz).all(axis=1)
    xyz = xyz[finite]
    if rgb is not None:
        rgb = rgb[finite]
    if len(xyz) < 100:
        raise ValueError(f"PLY file has too few finite vertices: {len(xyz)}")
    return xyz, rgb


def bbox_corners(cx: float, cy: float, width: float, depth: float, theta: float) -> np.ndarray:
    hw, hd = width / 2.0, depth / 2.0
    local = np.array([[-hw, -hd], [hw, -hd], [hw, hd], [-hw, hd]], dtype=np.float64)
    c, s = math.cos(theta), math.sin(theta)
    rot = np.array([[c, -s], [s, c]], dtype=np.float64)
    return local @ rot.T + np.array([cx, cy], dtype=np.float64)


def raster_polygon_from_points(
    xy: np.ndarray,
    fallback_xy: np.ndarray,
    resolution: float,
    close_iters: int = 1,
    dilate_iters: int = 0,
    simplify_m: float = 0.02,
    min_area: float = 0.25,
) -> Tuple[Polygon, Dict[str, object]]:
    """Build a non-convex footprint polygon from occupied XY cells.

    This is intentionally not a convex hull. It preserves L-shapes and inset
    corners much better, which matters for room planning.
    """
    debug: Dict[str, object] = {"method": "raster_occupancy_polygon"}
    if xy is None or len(xy) < 20:
        debug["fallback"] = "too_few_source_points_convex_hull"
        return MultiPoint(fallback_xy).convex_hull, debug

    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) < 20:
        debug["fallback"] = "too_few_finite_source_points_convex_hull"
        return MultiPoint(fallback_xy).convex_hull, debug

    # Trim extreme outliers before rasterization.
    lo = np.percentile(xy, 0.5, axis=0)
    hi = np.percentile(xy, 99.5, axis=0)
    keep = np.all((xy >= lo) & (xy <= hi), axis=1)
    if keep.sum() >= 20:
        xy = xy[keep]

    mn = xy.min(axis=0) - 2 * resolution
    mx = xy.max(axis=0) + 2 * resolution
    wh = np.maximum(np.ceil((mx - mn) / resolution).astype(int) + 1, 3)
    if int(wh[0] * wh[1]) > 10_000_000:
        debug["fallback"] = f"grid_too_large_{wh[0]}x{wh[1]}_convex_hull"
        return MultiPoint(xy).convex_hull, debug

    ij = np.floor((xy - mn) / resolution).astype(np.int32)
    ij[:, 0] = np.clip(ij[:, 0], 0, wh[0] - 1)
    ij[:, 1] = np.clip(ij[:, 1], 0, wh[1] - 1)
    grid = np.zeros((wh[1], wh[0]), dtype=bool)
    grid[ij[:, 1], ij[:, 0]] = True
    structure = np.ones((3, 3), dtype=bool)
    for _ in range(max(0, close_iters)):
        grid = binary_closing(grid, structure=structure)
    grid = binary_fill_holes(grid)
    for _ in range(max(0, dilate_iters)):
        grid = binary_dilation(grid, structure=structure)

    ys, xs = np.nonzero(grid)
    debug.update({"grid_width": int(wh[0]), "grid_height": int(wh[1]), "occupied_cells": int(len(xs))})
    if len(xs) == 0:
        debug["fallback"] = "empty_grid_convex_hull"
        return MultiPoint(xy).convex_hull, debug

    cells = [
        box(
            mn[0] + x * resolution,
            mn[1] + y * resolution,
            mn[0] + (x + 1) * resolution,
            mn[1] + (y + 1) * resolution,
        )
        for y, x in zip(ys, xs)
    ]
    poly = unary_union(cells).buffer(0)
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    if poly.is_empty or not isinstance(poly, Polygon) or poly.area < min_area:
        debug["fallback"] = "invalid_or_too_small_convex_hull"
        poly = MultiPoint(xy).convex_hull
    if simplify_m > 0:
        # Keep this smaller than the cell size so corners are not wiped out.
        poly = poly.simplify(simplify_m, preserve_topology=True)
    return poly, debug


def estimate_room_polygon(points: np.ndarray, args: argparse.Namespace) -> Tuple[Polygon, Dict[str, object]]:
    z = points[:, 2]
    floor_mask = z <= args.floor_height_threshold
    lower_mask = z <= args.room_slice_max_z

    if args.room_source == "floor":
        source_xy = points[floor_mask, :2]
        source_name = "floor"
    elif args.room_source == "lower_slice":
        source_xy = points[lower_mask, :2]
        source_name = f"lower_slice_z<={args.room_slice_max_z}"
    else:
        # The default uses lower slice first. If floor is hidden by furniture,
        # this still sees boundary evidence from low walls/furniture edges.
        source_xy = points[lower_mask, :2]
        source_name = f"auto_lower_slice_z<={args.room_slice_max_z}"
        if len(source_xy) < 100:
            source_xy = points[floor_mask, :2]
            source_name = "auto_floor_fallback"
    poly, dbg = raster_polygon_from_points(
        source_xy,
        fallback_xy=points[:, :2],
        resolution=args.room_resolution_m,
        close_iters=args.room_close_iters,
        dilate_iters=args.room_dilate_iters,
        simplify_m=args.room_simplify_m,
        min_area=args.min_room_area,
    )
    dbg["source"] = source_name
    dbg["source_points"] = int(len(source_xy))
    return poly, dbg


def grid_components_from_indices(
    points: np.ndarray,
    selected_idx: np.ndarray,
    resolution: float,
    close_iters: int,
    dilate_iters: int,
    min_cells: int,
) -> Tuple[List[np.ndarray], Dict[str, object]]:
    if len(selected_idx) == 0:
        return [], {"selected_points": 0, "components_raw": 0, "components_kept": 0}
    xy = points[selected_idx, :2]
    mn = xy.min(axis=0) - 2 * resolution
    mx = xy.max(axis=0) + 2 * resolution
    wh = np.maximum(np.ceil((mx - mn) / resolution).astype(int) + 1, 3)
    if int(wh[0] * wh[1]) > 12_000_000:
        raise RuntimeError(f"Object occupancy grid too large ({wh[0]}x{wh[1]}). Increase --occupancy_resolution.")
    ij = np.floor((xy - mn) / resolution).astype(np.int32)
    ij[:, 0] = np.clip(ij[:, 0], 0, wh[0] - 1)
    ij[:, 1] = np.clip(ij[:, 1], 0, wh[1] - 1)
    grid = np.zeros((wh[1], wh[0]), dtype=bool)
    grid[ij[:, 1], ij[:, 0]] = True
    structure = np.ones((3, 3), dtype=bool)
    for _ in range(max(0, close_iters)):
        grid = binary_closing(grid, structure=structure)
    for _ in range(max(0, dilate_iters)):
        grid = binary_dilation(grid, structure=structure)
    labeled, ncomp = cc_label(grid, structure=structure)
    point_component = labeled[ij[:, 1], ij[:, 0]]
    comps: List[np.ndarray] = []
    for cid in range(1, ncomp + 1):
        cell_count = int((labeled == cid).sum())
        if cell_count < min_cells:
            continue
        verts = selected_idx[point_component == cid]
        if len(verts):
            comps.append(verts.astype(np.int64))
    return comps, {
        "selected_points": int(len(selected_idx)),
        "grid_width": int(wh[0]),
        "grid_height": int(wh[1]),
        "components_raw": int(ncomp),
        "components_kept_by_cells": int(len(comps)),
    }


def robust_box_from_points(xy: np.ndarray, trim_pct: float = 1.0) -> Tuple[np.ndarray, float, float, float, np.ndarray]:
    if len(xy) >= 20 and trim_pct > 0:
        lo = np.percentile(xy, trim_pct, axis=0)
        hi = np.percentile(xy, 100 - trim_pct, axis=0)
        keep = np.all((xy >= lo) & (xy <= hi), axis=1)
        if keep.sum() >= max(10, 0.50 * len(xy)):
            xy = xy[keep]
    return oriented_bbox_2d(xy)


def room_boundary_distance(poly: Polygon, x: float, y: float) -> float:
    p = MultiPoint([[x, y]]).geoms[0]
    if poly.is_empty:
        return float("inf")
    return float(poly.exterior.distance(p))


def label_proposal(width: float, depth: float, height: float, z_min: float, z_max: float, band: str, room_poly: Polygon, center: np.ndarray) -> Tuple[str, float]:
    """Geometry-only semantic guess used for proposal scoring.

    Keep this conservative: the later v5 refinement pass is more detailed.  This
    function's main job is to avoid letting tiny/tall fragments dominate NMS and
    to stop table/bed/couch-sized proposals from being scored as cabinets.
    """
    width = float(max(width, 1e-6))
    depth = float(max(depth, 1e-6))
    area = width * depth
    long_side = max(width, depth)
    short_side = min(width, depth)
    aspect = long_side / max(short_side, 1e-6)
    boundary_dist = room_boundary_distance(room_poly, float(center[0]), float(center[1]))
    near_wall = boundary_dist <= 0.55

    # Large low/medium furniture first; clutter on top can make z_max high.
    if long_side >= 1.70 and short_side >= 0.72 and area >= 1.25 and z_min <= 0.55:
        if long_side >= 1.85 and short_side >= 0.82 and area >= 1.55:
            return "bed", 0.58
        return "sofa", 0.50

    if long_side >= 1.20 and short_side >= 0.38 and area >= 0.60 and z_min <= 0.65 and z_max <= 1.65:
        return "sofa", 0.40

    # Counter/desk/table surfaces before storage.
    if 0.28 <= area <= 3.80 and long_side >= 0.65 and short_side >= 0.22 and z_min <= 1.20 and z_max <= 1.85:
        if near_wall and long_side >= 1.05:
            if long_side >= 1.45 and short_side <= 0.95:
                return "kitchen_counter", 0.46
            return "desk", 0.42
        if band in {"tabletop", "countertop", "all"}:
            return "table", 0.42

    # Strict lamp/stand rule. This used to be too broad and created many false positives.
    if area <= 0.09 and long_side <= 0.42 and height >= 0.85 and z_min <= 0.35 and z_max >= 1.15:
        return "floor_lamp_or_tall_thin_object", 0.32

    # Chair/stool-like.
    if 0.10 <= area <= 0.65 and 0.20 <= z_max <= 1.35:
        return "chair_or_stool", 0.32

    # Tall storage only if not table/bed-sized.
    if height >= 1.25 and area >= 0.16:
        if aspect >= 1.40 or near_wall:
            return "cabinet_or_shelf", 0.30
        return "shelf_or_bookcase", 0.28

    if area >= 0.45 and height <= 1.45:
        if near_wall:
            return "desk", 0.25
        return "table", 0.24
    if area >= 0.18:
        return "furniture_proposal", 0.16
    return "small_object_proposal", 0.08


def proposal_score(label: str, area: float, height: float, point_count: int, band_name: str) -> float:
    label_bonus = {
        "bed": 2.0,
        "sofa": 1.8,
        "kitchen_counter": 1.7,
        "work_desk": 1.6,
        "desk": 1.5,
        "table": 1.5,
        "floor_lamp_or_tall_thin_object": 0.7,
        "chair_or_stool": 1.0,
        "cabinet_or_shelf": 1.1,
        "shelf_or_bookcase": 1.1,
        "furniture_proposal": 0.25,
    }.get(label, 0.25)
    band_bonus = {
        "low_large": 0.8,
        "tabletop": 0.8,
        "countertop": 0.8,
        "tall": 0.6,
        "all": 0.3,
    }.get(band_name, 0.0)
    return float(label_bonus + band_bonus + min(math.log1p(point_count) / 10.0, 0.9) + min(area, 2.0) * 0.10 + min(height, 2.0) * 0.05)


def build_proposals_for_band(
    points: np.ndarray,
    base_mask: np.ndarray,
    room_poly: Polygon,
    band_name: str,
    z_min: float,
    z_max: float,
    args: argparse.Namespace,
) -> Tuple[List[Proposal], Dict[str, object]]:
    z = points[:, 2]
    selected_idx = np.where(base_mask & (z >= z_min) & (z <= z_max))[0]
    comps, debug = grid_components_from_indices(
        points,
        selected_idx,
        resolution=args.occupancy_resolution,
        close_iters=args.object_close_iters,
        dilate_iters=args.object_dilate_iters,
        min_cells=args.min_component_cells,
    )
    debug.update({"band": band_name, "z_min": z_min, "z_max": z_max})

    proposals: List[Proposal] = []
    room_area = max(float(room_poly.area), 1e-6)
    for comp in comps:
        if len(comp) < args.min_object_points:
            continue
        pts = points[comp]
        xy = pts[:, :2]
        center, width, depth, theta, corners = robust_box_from_points(xy, trim_pct=args.box_trim_percentile)
        width = float(max(width + 2 * args.box_padding, 0.04))
        depth = float(max(depth + 2 * args.box_padding, 0.04))
        corners = bbox_corners(float(center[0]), float(center[1]), width, depth, float(theta))
        zlo = float(np.percentile(pts[:, 2], 1.0))
        zhi = float(np.percentile(pts[:, 2], 99.0))
        height = max(0.0, zhi - zlo)
        area = width * depth
        long_side = max(width, depth)

        reject = None
        if len(comp) < args.min_object_points:
            reject = "min_points"
        elif height < args.min_object_height:
            reject = "min_height"
        elif height > args.max_object_height:
            reject = "max_height"
        elif area < args.min_box_area:
            reject = "min_area"
        elif area > args.max_box_area:
            reject = "max_area_absolute"
        elif area / room_area > args.max_box_area_ratio:
            reject = "max_area_ratio"
        elif long_side > args.max_box_long_side_ratio * max(math.sqrt(room_area), 1e-6) and area / room_area > 0.18:
            reject = "room_sized_component"

        hull = MultiPoint(xy).convex_hull if len(xy) >= 3 else MultiPoint(xy).convex_hull
        outside_dist = float(room_poly.distance(hull)) if not hull.is_empty else 0.0
        if args.filter_outside_room and outside_dist > args.outside_room_tolerance:
            reject = "outside_room"

        label, confidence = label_proposal(width, depth, height, zlo, zhi, band_name, room_poly, center)
        score = proposal_score(label, area, height, len(comp), band_name) + confidence
        proposals.append(Proposal(
            vertex_ids=comp,
            band_name=band_name,
            center=center,
            width=width,
            depth=depth,
            theta=theta,
            corners=corners,
            z_min=zlo,
            z_max=zhi,
            height=height,
            label=label,
            score=score,
            reject_reason=reject,
        ))
    debug["proposals_total"] = int(len(proposals))
    debug["proposals_kept_pre_nms"] = int(sum(1 for p in proposals if p.reject_reason is None))
    debug["proposals_rejected"] = int(sum(1 for p in proposals if p.reject_reason is not None))
    return proposals, debug


def polygon_iou(a: Polygon, b: Polygon) -> float:
    if a.is_empty or b.is_empty:
        return 0.0
    inter = a.intersection(b).area
    union = a.union(b).area
    if union <= 0:
        return 0.0
    return float(inter / union)


def nms_proposals(proposals: List[Proposal], iou_threshold: float, contain_threshold: float) -> List[Proposal]:
    candidates = [p for p in proposals if p.reject_reason is None]
    candidates.sort(key=lambda p: p.score, reverse=True)
    kept: List[Proposal] = []
    for p in candidates:
        pp = p.poly
        discard = False
        for q in kept:
            qq = q.poly
            iou = polygon_iou(pp, qq)
            inter = pp.intersection(qq).area if (not pp.is_empty and not qq.is_empty) else 0.0
            containment = inter / max(min(pp.area, qq.area), 1e-6)
            if iou >= iou_threshold or containment >= contain_threshold:
                discard = True
                break
        if not discard:
            kept.append(p)
    return kept


def make_base_candidate_mask(points: np.ndarray, floor_mask: np.ndarray, wall_mask: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    z = points[:, 2]
    base = (z >= args.object_min_z) & (z <= args.object_max_z) & (~floor_mask)
    if args.wall_filter == "none":
        return base
    if args.wall_filter == "high_only":
        return base & (~(wall_mask & (z >= args.wall_filter_min_z)))
    if args.wall_filter == "hard":
        return base & (~wall_mask)
    raise ValueError(f"Unknown wall_filter: {args.wall_filter}")


def build_layout(args: argparse.Namespace) -> Tuple[SceneLayout, Dict[str, object], np.ndarray, Optional[np.ndarray], Dict[str, object]]:
    rng = np.random.default_rng(args.seed)
    ply = Path(args.ply)
    raw_xyz, rgb = load_user_ply_vertices(ply)

    if args.assume_aligned:
        aligned = raw_xyz.astype(np.float32).copy()
        floor_z = float(np.percentile(aligned[:, 2], 1.0))
        aligned[:, 2] -= floor_z
        axis_alignment = np.eye(4, dtype=np.float32)
        axis_alignment[2, 3] = -floor_z
        align_meta = {"mode": "assume_aligned", "floor_z_translation": float(-floor_z)}
    else:
        aligned, axis_alignment, align_meta = estimate_alignment(
            raw_xyz,
            up_axis=args.up_axis,
            plane_distance=args.plane_distance,
            ransac_iterations=args.ransac_iterations,
            rng=rng,
        )

    z = aligned[:, 2]
    floor_mask = z <= args.floor_height_threshold
    ceiling_mask = z >= args.ceiling_min_z if args.ceiling_min_z > 0 else np.zeros(len(aligned), dtype=bool)

    wall_base = (z > args.floor_height_threshold) & (z < args.wall_detect_max_z)
    wall_planes, wall_mask = detect_wall_planes(
        aligned,
        base_mask=wall_base,
        plane_distance=max(args.plane_distance * 1.5, 0.04),
        ransac_iterations=args.ransac_iterations,
        rng=rng,
    )

    room_poly, room_debug = estimate_room_polygon(aligned, args)
    base_mask = make_base_candidate_mask(aligned, floor_mask | ceiling_mask, wall_mask, args)

    # Multi-pass proposals. The bands intentionally overlap. NMS merges duplicates.
    bands = [
        ("low_large",  args.object_min_z, min(args.low_band_max_z, args.object_max_z)),
        ("tabletop",   max(args.object_min_z, args.table_band_min_z), min(args.table_band_max_z, args.object_max_z)),
        ("countertop", max(args.object_min_z, args.counter_band_min_z), min(args.counter_band_max_z, args.object_max_z)),
        ("tall",       max(args.object_min_z, args.tall_band_min_z), min(args.object_max_z, args.tall_band_max_z)),
        ("all",        args.object_min_z, args.object_max_z),
    ]

    all_proposals: List[Proposal] = []
    band_debug: List[Dict[str, object]] = []
    for band_name, zmin, zmax in bands:
        if zmax <= zmin:
            continue
        props, dbg = build_proposals_for_band(aligned, base_mask, room_poly, band_name, zmin, zmax, args)
        all_proposals.extend(props)
        band_debug.append(dbg)

    kept = nms_proposals(all_proposals, iou_threshold=args.nms_iou, contain_threshold=args.nms_containment)
    kept = kept[: args.max_objects]

    seg_indices = np.zeros(len(aligned), dtype=np.int32)
    semantic_labels = np.full(len(aligned), PSEUDO_SEMANTIC_ID["unlabeled"], dtype=np.int32)
    semantic_labels[floor_mask] = PSEUDO_SEMANTIC_ID["floor"]
    semantic_labels[wall_mask] = PSEUDO_SEMANTIC_ID["wall"]
    semantic_labels[ceiling_mask] = PSEUDO_SEMANTIC_ID.get("ceiling", 22)
    seg_indices[floor_mask] = 0
    seg_indices[wall_mask] = 1
    seg_indices[ceiling_mask] = 2

    objects: List[Object2D] = []
    seg_groups: List[dict] = []
    next_seg = 10
    for obj_id, p in enumerate(kept, start=1):
        sem_id = PSEUDO_SEMANTIC_ID.get(p.label, PSEUDO_SEMANTIC_ID["unknown_object"])
        seg_indices[p.vertex_ids] = next_seg
        semantic_labels[p.vertex_ids] = sem_id
        pts = aligned[p.vertex_ids]
        center_3d = pts.mean(axis=0)
        size_3d = pts.max(axis=0) - pts.min(axis=0)
        objects.append(Object2D(
            id=obj_id,
            object_id=obj_id,
            label=p.label,
            raw_label=f"{p.label}__pseudo_from_{p.band_name}",
            cx=float(p.center[0]),
            cy=float(p.center[1]),
            width=float(p.width),
            depth=float(p.depth),
            theta=float(p.theta),
            z_min=float(p.z_min),
            z_max=float(p.z_max),
            height=float(p.height),
            point_count=int(len(p.vertex_ids)),
            footprint=[[float(x), float(y)] for x, y in p.corners.tolist()],
            bbox3d_center=[float(v) for v in center_3d.tolist()],
            bbox3d_size=[float(v) for v in size_3d.tolist()],
        ))
        seg_groups.append({"objectId": int(obj_id), "id": int(obj_id), "label": p.label, "segments": [int(next_seg)]})
        next_seg += 1

    # Stable visual order: larger items first in the editor/list.
    objects.sort(key=lambda o: (-(o.width * o.depth), o.label.lower(), o.id))

    scene_id = args.scene_id or ply.stem.replace(" ", "_")
    metadata = {
        "source": "user_ply_geometry_pseudo_scannet_common",
        "input_ply": str(ply),
        "axis_alignment_estimated": not args.assume_aligned,
        "axis_alignment_matrix": [float(x) for x in axis_alignment.reshape(-1).tolist()],
        "alignment": align_meta,
        "proposal_mode": "multi_height_band_xy_occupancy_nms",
        "room_polygon_method": room_debug,
        "wall_filter": args.wall_filter,
        "floor_height_threshold": float(args.floor_height_threshold),
        "object_min_z": float(args.object_min_z),
        "object_max_z": float(args.object_max_z),
        "occupancy_resolution": float(args.occupancy_resolution),
        "raw_vertex_count": int(len(raw_xyz)),
        "floor_vertex_count": int(floor_mask.sum()),
        "detected_wall_vertex_count": int(wall_mask.sum()),
        "base_candidate_vertex_count": int(base_mask.sum()),
        "candidate_vertex_ratio": float(base_mask.mean()),
        "proposal_count_total": int(len(all_proposals)),
        "proposal_count_rejected_before_nms": int(sum(1 for p in all_proposals if p.reject_reason is not None)),
        "proposal_count_after_nms": int(len(kept)),
        "object_count": int(len(objects)),
        "labeling_method": "geometry-only semantic guesses; labels are pseudo labels and should be user-correctable or replaced by ML segmentation",
        "important_limitations": [
            "Raw user PLY files do not contain furniture labels or object instances.",
            "This script produces high-recall geometry proposals, not true object recognition.",
            "For reliable bed/desk/sofa/counter labels, replace this stage with ScanNet-trained 3D instance segmentation or 2D/3D multi-view labeling.",
        ],
    }
    layout = SceneLayout(
        scene_id=scene_id,
        units="meters",
        room_polygon=polygon_to_coords(room_poly),
        room_bbox=polygon_bbox(room_poly),
        objects=objects,
        metadata=metadata,
    )
    pseudo = {
        "axis_alignment": axis_alignment,
        "floor_mask": floor_mask,
        "wall_mask": wall_mask,
        "ceiling_mask": ceiling_mask,
        "candidate_mask": base_mask,
        "seg_indices": seg_indices,
        "semantic_labels": semantic_labels,
        "seg_groups": seg_groups,
        "wall_planes": [{"normal": [float(x) for x in n.tolist()], "d": float(d)} for n, d in wall_planes],
    }
    rejected_summary: Dict[str, int] = {}
    for p in all_proposals:
        if p.reject_reason:
            rejected_summary[p.reject_reason] = rejected_summary.get(p.reject_reason, 0) + 1
    debug = {
        "band_debug": band_debug,
        "room_debug": room_debug,
        "proposals": [
            {
                "band": p.band_name,
                "label": p.label,
                "score": p.score,
                "points": int(len(p.vertex_ids)),
                "cx": float(p.center[0]),
                "cy": float(p.center[1]),
                "width": float(p.width),
                "depth": float(p.depth),
                "height": float(p.height),
                "area": float(p.area),
                "reject_reason": p.reject_reason,
            }
            for p in all_proposals
        ],
        "rejected_summary": rejected_summary,
        "kept_proposals": [
            {
                "label": p.label,
                "band": p.band_name,
                "score": p.score,
                "points": int(len(p.vertex_ids)),
                "width": float(p.width),
                "depth": float(p.depth),
                "height": float(p.height),
                "area": float(p.area),
            }
            for p in kept
        ],
        "wall_planes": pseudo["wall_planes"],
    }
    return layout, pseudo, aligned, rgb, debug


def write_pseudo_scannet_files(out_dir: str | Path, scene_id: str, input_ply: str | Path, aligned: np.ndarray, rgb: Optional[np.ndarray], pseudo: Dict[str, object], copy_original: bool = True) -> Path:
    scene_dir = Path(out_dir) / "pseudo_scannet" / "scans" / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    write_xyzrgb_ply(scene_dir / f"{scene_id}_vh_clean_2.ply", aligned, rgb)
    with (scene_dir / f"{scene_id}_vh_clean_2.0.010000.segs.json").open("w", encoding="utf-8") as f:
        json.dump({"sceneId": scene_id, "segIndices": [int(x) for x in np.asarray(pseudo["seg_indices"]).tolist()]}, f)
    with (scene_dir / f"{scene_id}.aggregation.json").open("w", encoding="utf-8") as f:
        json.dump({"sceneId": scene_id, "appId": "user-ply-pseudo-scannet-common", "segGroups": pseudo["seg_groups"]}, f, indent=2)
    write_labels_ply(scene_dir / f"{scene_id}_vh_clean_2.labels.ply", np.asarray(pseudo["semantic_labels"]))
    axis = np.asarray(pseudo["axis_alignment"], dtype=np.float32)
    with (scene_dir / f"{scene_id}.txt").open("w", encoding="utf-8") as f:
        f.write(f"sceneId = {scene_id}\n")
        f.write("source = user_ply_pseudo_scannet_common\n")
        f.write("axisAlignment = " + " ".join(f"{float(x):.8f}" for x in axis.reshape(-1)) + "\n")
    with (scene_dir / f"{scene_id}.pseudo_debug.json").open("w", encoding="utf-8") as f:
        json.dump({
            "source_input_ply": str(input_ply),
            "note": "Generated from raw user geometry. Labels are pseudo labels, not ScanNet human annotations.",
            "wall_planes": pseudo.get("wall_planes", []),
            "segment_meanings": {"0": "floor_or_unlabeled", "1": "wall", "2": "ceiling", "10+": "object proposal segments"},
        }, f, indent=2)
    if copy_original:
        try:
            shutil.copy2(Path(input_ply), scene_dir / f"{scene_id}.original_input.ply")
        except Exception:
            pass
    return scene_dir


def write_debug(out_dir: str | Path, layout: SceneLayout, aligned: np.ndarray, pseudo: Dict[str, object], debug: Dict[str, object], seed: int) -> None:
    out_dir = Path(out_dir)
    with (out_dir / "debug_detection_summary.json").open("w", encoding="utf-8") as f:
        json.dump({"metadata": layout.metadata, **debug}, f, indent=2)

    rng = np.random.default_rng(seed)
    n = len(aligned)
    idx = rng.choice(n, size=min(n, 160000), replace=False) if n > 160000 else np.arange(n)
    pts = aligned[idx]
    floor = np.asarray(pseudo["floor_mask"])[idx]
    wall = np.asarray(pseudo["wall_mask"])[idx]
    cand = np.asarray(pseudo["candidate_mask"])[idx]

    fig, ax = plt.subplots(figsize=(11, 11))
    ax.scatter(pts[:, 0], pts[:, 1], s=0.18, alpha=0.06, label="all sampled")
    if floor.any():
        p = pts[floor]
        ax.scatter(p[:, 0], p[:, 1], s=0.30, alpha=0.20, label="floor/low")
    if wall.any():
        p = pts[wall]
        ax.scatter(p[:, 0], p[:, 1], s=0.30, alpha=0.16, label="detected wall planes")
    if cand.any():
        p = pts[cand]
        ax.scatter(p[:, 0], p[:, 1], s=0.38, alpha=0.35, label="object candidate points")

    room = np.asarray(layout.room_polygon, dtype=float)
    if len(room) >= 3:
        closed = np.vstack([room, room[0]])
        ax.plot(closed[:, 0], closed[:, 1], linewidth=2.0, label="room polygon")
    for obj in layout.objects:
        corners = np.asarray(obj.footprint, dtype=float)
        closed = np.vstack([corners, corners[0]])
        ax.plot(closed[:, 0], closed[:, 1], linewidth=1.5)
        ax.text(obj.cx, obj.cy, f"{obj.id}:{obj.label}", fontsize=7)
    ax.axis("equal")
    ax.legend(loc="best", markerscale=8)
    ax.set_title("Debug layers: room footprint, candidate points, exported object boxes")
    fig.tight_layout()
    fig.savefig(out_dir / "debug_layers.png", dpi=180)
    plt.close(fig)

    # Proposal-only debug plot helps diagnose why one large component was rejected.
    fig, ax = plt.subplots(figsize=(11, 11))
    if len(room) >= 3:
        closed = np.vstack([room, room[0]])
        ax.plot(closed[:, 0], closed[:, 1], linewidth=2.0, label="room polygon")
    for pr in debug.get("proposals", []):
        x, y, w, d = pr["cx"], pr["cy"], pr["width"], pr["depth"]
        # Use axis-aligned marker for debug only; final layout uses true footprint.
        alpha = 0.18 if pr.get("reject_reason") else 0.65
        ax.add_patch(plt.Rectangle((x - w / 2, y - d / 2), w, d, fill=False, linewidth=1.0, alpha=alpha))
        if not pr.get("reject_reason"):
            ax.text(x, y, pr["label"], fontsize=6)
    ax.axis("equal")
    ax.set_title("All proposals before NMS/filters (faint = rejected)")
    fig.tight_layout()
    fig.savefig(out_dir / "debug_proposals.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Raw user .ply -> pseudo-ScanNet + top-down draggable layout exporter shared geometry")
    p.add_argument("--ply", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--scene_id", default=None)

    # Alignment
    p.add_argument("--up_axis", default="z", choices=["auto", "x", "y", "z"])
    p.add_argument("--assume_aligned", action="store_true")
    p.add_argument("--plane_distance", type=float, default=0.04)
    p.add_argument("--ransac_iterations", type=int, default=900)

    # Room polygon
    p.add_argument("--floor_height_threshold", type=float, default=0.06)
    p.add_argument("--room_source", default="auto", choices=["auto", "floor", "lower_slice"])
    p.add_argument("--room_slice_max_z", type=float, default=0.35)
    p.add_argument("--room_resolution_m", type=float, default=0.05)
    p.add_argument("--room_close_iters", type=int, default=1)
    p.add_argument("--room_dilate_iters", type=int, default=0)
    p.add_argument("--room_simplify_m", type=float, default=0.015)
    p.add_argument("--min_room_area", type=float, default=0.50)

    # Structural filtering
    p.add_argument("--wall_filter", default="high_only", choices=["none", "high_only", "hard"])
    p.add_argument("--wall_filter_min_z", type=float, default=1.35)
    p.add_argument("--wall_detect_max_z", type=float, default=2.40)
    p.add_argument("--ceiling_min_z", type=float, default=0.0, help="Set >0 if ceiling points should be excluded, e.g. 2.4")

    # Object proposal bands
    p.add_argument("--object_min_z", type=float, default=0.08)
    p.add_argument("--object_max_z", type=float, default=2.15)
    p.add_argument("--low_band_max_z", type=float, default=0.85)
    p.add_argument("--table_band_min_z", type=float, default=0.42)
    p.add_argument("--table_band_max_z", type=float, default=1.20)
    p.add_argument("--counter_band_min_z", type=float, default=0.70)
    p.add_argument("--counter_band_max_z", type=float, default=1.35)
    p.add_argument("--tall_band_min_z", type=float, default=1.00)
    p.add_argument("--tall_band_max_z", type=float, default=2.15)

    # Object proposal clustering/filtering
    p.add_argument("--occupancy_resolution", type=float, default=0.045)
    p.add_argument("--object_close_iters", type=int, default=0)
    p.add_argument("--object_dilate_iters", type=int, default=0)
    p.add_argument("--min_component_cells", type=int, default=1)
    p.add_argument("--min_object_points", type=int, default=12)
    p.add_argument("--min_object_height", type=float, default=0.035)
    p.add_argument("--max_object_height", type=float, default=2.40)
    p.add_argument("--min_box_area", type=float, default=0.004)
    p.add_argument("--max_box_area", type=float, default=8.0)
    p.add_argument("--max_box_area_ratio", type=float, default=0.32)
    p.add_argument("--max_box_long_side_ratio", type=float, default=0.90)
    p.add_argument("--box_padding", type=float, default=0.08)
    p.add_argument("--box_trim_percentile", type=float, default=1.0)
    p.add_argument("--filter_outside_room", action="store_true")
    p.add_argument("--outside_room_tolerance", type=float, default=0.30)

    # Merge/sort
    p.add_argument("--nms_iou", type=float, default=0.45)
    p.add_argument("--nms_containment", type=float, default=0.72)
    p.add_argument("--max_objects", type=int, default=80)

    # Output
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--no_png", action="store_true")
    p.add_argument("--no_debug", action="store_true")
    p.add_argument("--no_copy_original", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layout, pseudo, aligned, rgb, debug = build_layout(args)
    save_layout_json(layout, out_dir / "scene_layout.json")
    if not args.no_png:
        render_layout_png(layout, out_dir / "scene_layout.png")
    scene_id = args.scene_id or Path(args.ply).stem.replace(" ", "_")
    pseudo_dir = write_pseudo_scannet_files(out_dir, scene_id, args.ply, aligned, rgb, pseudo, copy_original=not args.no_copy_original)
    if not args.no_debug:
        write_debug(out_dir, layout, aligned, pseudo, debug, args.seed)

    print(f"Saved layout JSON to: {out_dir / 'scene_layout.json'}")
    if not args.no_png:
        print(f"Saved layout preview to: {out_dir / 'scene_layout.png'}")
    if not args.no_debug:
        print(f"Saved debug summary to: {out_dir / 'debug_detection_summary.json'}")
        print(f"Saved debug layers image to: {out_dir / 'debug_layers.png'}")
        print(f"Saved proposal debug image to: {out_dir / 'debug_proposals.png'}")
    print(f"Saved pseudo-ScanNet scene folder to: {pseudo_dir}")
    print(f"Objects exported: {len(layout.objects)}")


if __name__ == "__main__":
    main()
