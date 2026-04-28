#!/usr/bin/env python3
"""
Export an arbitrary user .ply room scan into the same top-down layout format used
by export_scannet_topdown.py, while also writing ScanNet-like pseudo annotation
files for debugging / later training.

Unlike ScanNet, a user .ply usually does NOT contain:
- <scene>.aggregation.json
- <scene>_vh_clean_2.0.010000.segs.json
- <scene>_vh_clean_2.labels.ply
- <scene>.txt with axisAlignment

This script estimates replacement annotations from geometry:
1. Load raw .ply vertices.
2. Estimate a floor plane with RANSAC.
3. Align the floor to z=0, giving a ScanNet-like axisAlignment transform.
4. Remove floor and large wall-like planes.
5. Cluster the remaining points into object candidates.
6. Assign rough furniture labels from box geometry.
7. Write pseudo ScanNet-style files and the same scene_layout.json/png used by
   the browser room_editor.html.

This is meant as an MVP / bridge stage. It creates labeled, draggable boxes, but
it is not a replacement for a trained 3D semantic/instance segmentation model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint, Polygon

from export_scannet_topdown import (
    Object2D,
    SceneLayout,
    apply_transform,
    oriented_bbox_2d,
    polygon_bbox,
    polygon_to_coords,
    read_vertex_table_from_ply,
    render_layout_png,
    room_polygon_from_floor_points,
    save_layout_json,
)


# Rough NYU40-ish IDs. These only exist so the generated labels .ply has a
# semantic label column similar to ScanNet's labels .ply. The editor mainly uses
# the aggregation labels and scene_layout.json labels.
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


# ----------------------------
# Basic PLY helpers
# ----------------------------

def load_user_ply_vertices(path: str | Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Read xyz and optional rgb from a user PLY vertex table."""
    props = read_vertex_table_from_ply(path, None)
    for key in ("x", "y", "z"):
        if key not in props:
            raise ValueError(f"PLY file is missing vertex property '{key}': {path}")

    xyz = np.column_stack([props["x"], props["y"], props["z"]]).astype(np.float32)

    rgb = None
    # Common property names are red/green/blue; some exporters use r/g/b.
    if all(k in props for k in ("red", "green", "blue")):
        rgb = np.column_stack([props["red"], props["green"], props["blue"]]).astype(np.uint8)
    elif all(k in props for k in ("r", "g", "b")):
        rgb = np.column_stack([props["r"], props["g"], props["b"]]).astype(np.uint8)

    finite = np.isfinite(xyz).all(axis=1)
    if not np.all(finite):
        xyz = xyz[finite]
        if rgb is not None:
            rgb = rgb[finite]

    if len(xyz) < 100:
        raise ValueError(f"PLY file has too few finite vertices to process: {len(xyz)}")

    return xyz, rgb


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


# ----------------------------
# Geometry utilities
# ----------------------------

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


def connected_components_radius(points: np.ndarray, eps: float, min_points: int) -> List[np.ndarray]:
    """Simple radius graph connected components on downsampled points.

    This is intentionally more forgiving than full DBSCAN. It groups nearby
    remaining furniture points into blobs, then filters out small components.
    """
    if len(points) == 0:
        return []
    tree = cKDTree(points)
    visited = np.zeros(len(points), dtype=bool)
    comps: List[np.ndarray] = []

    for start in range(len(points)):
        if visited[start]:
            continue
        q: deque[int] = deque([start])
        visited[start] = True
        comp = []
        while q:
            i = q.popleft()
            comp.append(i)
            neigh = tree.query_ball_point(points[i], r=eps)
            for j in neigh:
                if not visited[j]:
                    visited[j] = True
                    q.append(j)
        if len(comp) >= min_points:
            comps.append(np.asarray(comp, dtype=np.int64))
    return comps


def rough_label_from_dimensions(width: float, depth: float, height: float, z_min: float, z_max: float) -> str:
    """Assign a coarse label from geometry. Tuned to be honest but useful."""
    w, d = sorted([float(width), float(depth)], reverse=True)
    area = w * d

    if area >= 1.35 and height <= 1.25 and z_max <= 1.4:
        return "bed_or_sofa"
    if area >= 0.6 and 0.35 <= height <= 1.25 and z_max <= 1.5:
        return "table_or_desk"
    if area < 0.9 and 0.45 <= height <= 1.6:
        return "chair_or_small_furniture"
    if height >= 1.4 and area >= 0.25:
        return "shelf_or_cabinet"
    if area >= 1.2:
        return "large_furniture"
    if area >= 0.15:
        return "small_furniture"
    return "unknown_object"


def clean_component_points(points: np.ndarray, low_q: float = 1.0, high_q: float = 99.0) -> np.ndarray:
    """Trim extreme outliers within a component before boxing."""
    if len(points) < 20:
        return points
    lo = np.percentile(points, low_q, axis=0)
    hi = np.percentile(points, high_q, axis=0)
    mask = np.all((points >= lo) & (points <= hi), axis=1)
    if mask.sum() < max(10, 0.5 * len(points)):
        return points
    return points[mask]


# ----------------------------
# Main conversion
# ----------------------------

def build_layout_from_user_ply(
    ply_path: str | Path,
    scene_id: str,
    voxel_size: float,
    plane_distance: float,
    floor_height_threshold: float,
    object_min_z: float,
    cluster_eps: float,
    cluster_min_points: int,
    min_object_points: int,
    min_object_height: float,
    max_object_height: float,
    room_resolution_m: float,
    up_axis: str,
    ransac_iterations: int,
    seed: int,
    include_structural: bool,
) -> Tuple[SceneLayout, Dict[str, object], np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray, List[dict]]:
    rng = np.random.default_rng(seed)
    raw_xyz, rgb = load_user_ply_vertices(ply_path)

    aligned_xyz, axis_alignment, align_meta = estimate_alignment(
        raw_xyz,
        up_axis=up_axis,
        plane_distance=plane_distance,
        ransac_iterations=ransac_iterations,
        rng=rng,
    )

    floor_mask = aligned_xyz[:, 2] <= floor_height_threshold

    # Candidate wall search: above the floor, not extreme ceiling/scan noise.
    z95 = float(np.percentile(aligned_xyz[:, 2], 95))
    wall_base_mask = (aligned_xyz[:, 2] > floor_height_threshold) & (aligned_xyz[:, 2] < max(z95, 1.0))
    wall_planes, wall_mask = detect_wall_planes(
        aligned_xyz,
        base_mask=wall_base_mask,
        plane_distance=max(plane_distance * 1.5, 0.04),
        ransac_iterations=ransac_iterations,
        rng=rng,
    )

    structural_mask = floor_mask | wall_mask
    ceiling_mask = np.zeros(len(aligned_xyz), dtype=bool)
    if z95 > 1.8:
        # Very rough ceiling/noise filter near the top of the scan. We do not use
        # this for room footprint, only to avoid furniture clusters at ceiling.
        ceiling_mask = aligned_xyz[:, 2] >= np.percentile(aligned_xyz[:, 2], 99.5)
        structural_mask |= ceiling_mask

    object_candidate_mask = (~structural_mask) & (aligned_xyz[:, 2] >= object_min_z)

    # Downsample only the candidate points for clustering.
    candidate_idx = np.where(object_candidate_mask)[0]
    candidate_points = aligned_xyz[candidate_idx]
    if len(candidate_points) > 0:
        rep_local = voxel_downsample_indices(candidate_points, voxel_size)
        cluster_points = candidate_points[rep_local]
        cluster_orig_idx = candidate_idx[rep_local]
    else:
        cluster_points = np.empty((0, 3), dtype=np.float32)
        cluster_orig_idx = np.empty((0,), dtype=np.int64)

    comps = connected_components_radius(cluster_points, eps=cluster_eps, min_points=cluster_min_points)

    # Per-original-vertex pseudo labels and segment IDs.
    seg_indices = np.zeros(len(aligned_xyz), dtype=np.int32)  # 0 floor/unlabeled by default
    semantic_labels = np.full(len(aligned_xyz), SEMANTIC_ID["unlabeled"], dtype=np.int32)
    semantic_labels[floor_mask] = SEMANTIC_ID["floor"]
    semantic_labels[wall_mask] = SEMANTIC_ID["wall"]
    semantic_labels[ceiling_mask] = SEMANTIC_ID["ceiling"]
    seg_indices[floor_mask] = 0
    seg_indices[wall_mask] = 1
    seg_indices[ceiling_mask] = 2

    floor_xy = aligned_xyz[floor_mask, :2]
    room_poly = room_polygon_from_floor_points(
        floor_xy=floor_xy,
        fallback_xy=aligned_xyz[:, :2],
        resolution=room_resolution_m,
    )

    objects: List[Object2D] = []
    seg_groups: List[dict] = []
    next_obj_id = 1
    next_seg_id = 10

    # Assign each original candidate point to the nearest downsampled clustering point.
    # This lets us expand boxes/segments from downsampled components back to original vertices.
    if len(cluster_points) > 0:
        cluster_tree = cKDTree(cluster_points)
        _, nearest_cluster_local = cluster_tree.query(candidate_points, k=1)
        nearest_cluster_local = nearest_cluster_local.astype(np.int64)
    else:
        nearest_cluster_local = np.empty((0,), dtype=np.int64)

    for comp in comps:
        # Expand downsampled component to original candidate vertices by nearest rep.
        comp_set = set(int(i) for i in comp.tolist())
        local_candidate_mask = np.array([int(i) in comp_set for i in nearest_cluster_local], dtype=bool)
        vertex_ids = candidate_idx[local_candidate_mask]
        if len(vertex_ids) < min_object_points:
            continue

        obj_points = aligned_xyz[vertex_ids]
        obj_points_clean = clean_component_points(obj_points)
        if len(obj_points_clean) < min_object_points:
            obj_points_clean = obj_points

        xy = obj_points_clean[:, :2]
        center_xy, width, depth, theta, corners = oriented_bbox_2d(xy)
        z_min = float(np.percentile(obj_points_clean[:, 2], 1))
        z_max = float(np.percentile(obj_points_clean[:, 2], 99))
        height = max(0.0, z_max - z_min)

        if height < min_object_height or height > max_object_height:
            continue
        if max(width, depth) < 0.10 or width * depth < 0.03:
            continue

        hull = MultiPoint(xy).convex_hull
        if not room_poly.buffer(0.30).contains(hull.centroid):
            if room_poly.distance(hull) > 0.50:
                continue

        label = rough_label_from_dimensions(width, depth, height, z_min, z_max)
        seg_id = next_seg_id
        object_id = next_obj_id
        semantic_id = SEMANTIC_ID.get(label, SEMANTIC_ID["unknown_object"])

        seg_indices[vertex_ids] = seg_id
        semantic_labels[vertex_ids] = semantic_id

        center_3d = obj_points_clean.mean(axis=0)
        size_3d = obj_points_clean.max(axis=0) - obj_points_clean.min(axis=0)

        objects.append(
            Object2D(
                id=next_obj_id,
                object_id=object_id,
                label=label,
                raw_label=label,
                cx=float(center_xy[0]),
                cy=float(center_xy[1]),
                width=float(max(width, 0.05)),
                depth=float(max(depth, 0.05)),
                theta=float(theta),
                z_min=float(z_min),
                z_max=float(z_max),
                height=float(height),
                point_count=int(len(vertex_ids)),
                footprint=[[float(x), float(y)] for x, y in corners.tolist()],
                bbox3d_center=[float(v) for v in center_3d.tolist()],
                bbox3d_size=[float(v) for v in size_3d.tolist()],
            )
        )
        seg_groups.append(
            {
                "objectId": int(object_id),
                "id": int(object_id),
                "label": label,
                "segments": [int(seg_id)],
            }
        )
        next_obj_id += 1
        next_seg_id += 1

    if include_structural:
        # Structural segments are useful for debugging but intentionally not
        # added as draggable objects.
        pass

    objects.sort(key=lambda o: (o.label.lower(), o.id))

    layout = SceneLayout(
        scene_id=scene_id,
        units="meters",
        room_polygon=polygon_to_coords(room_poly),
        room_bbox=polygon_bbox(room_poly),
        objects=objects,
        metadata={
            "source": "user_ply_geometry_pseudo_scannet",
            "input_ply": str(ply_path),
            "axis_alignment_estimated": True,
            "axis_alignment_matrix": [float(x) for x in axis_alignment.reshape(-1).tolist()],
            "alignment": align_meta,
            "voxel_size": float(voxel_size),
            "plane_distance": float(plane_distance),
            "floor_height_threshold": float(floor_height_threshold),
            "object_min_z": float(object_min_z),
            "cluster_eps": float(cluster_eps),
            "cluster_min_points": int(cluster_min_points),
            "min_object_points": int(min_object_points),
            "min_object_height": float(min_object_height),
            "max_object_height": float(max_object_height),
            "room_resolution_m": float(room_resolution_m),
            "raw_vertex_count": int(len(raw_xyz)),
            "object_candidate_vertex_count": int(object_candidate_mask.sum()),
            "floor_vertex_count": int(floor_mask.sum()),
            "wall_vertex_count": int(wall_mask.sum()),
            "wall_planes_detected": int(len(wall_planes)),
            "object_count": int(len(objects)),
            "labeling_method": "geometry heuristics; replaceable with ML segmentation later",
        },
    )

    pseudo = {
        "axis_alignment": axis_alignment,
        "floor_mask": floor_mask,
        "wall_mask": wall_mask,
        "ceiling_mask": ceiling_mask,
        "seg_indices": seg_indices,
        "semantic_labels": semantic_labels,
        "seg_groups": seg_groups,
        "wall_planes": [
            {"normal": [float(x) for x in n.tolist()], "d": float(d)}
            for n, d in wall_planes
        ],
    }
    return layout, pseudo, aligned_xyz, rgb, seg_indices, semantic_labels, seg_groups


def write_pseudo_scannet_files(
    out_dir: str | Path,
    scene_id: str,
    input_ply: str | Path,
    aligned_xyz: np.ndarray,
    rgb: Optional[np.ndarray],
    pseudo: Dict[str, object],
    copy_original: bool = True,
) -> Path:
    """Write a scene folder with ScanNet-like filenames."""
    out_dir = Path(out_dir)
    scene_dir = out_dir / "pseudo_scannet" / "scans" / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = scene_dir / f"{scene_id}_vh_clean_2.ply"
    seg_path = scene_dir / f"{scene_id}_vh_clean_2.0.010000.segs.json"
    agg_path = scene_dir / f"{scene_id}.aggregation.json"
    labels_path = scene_dir / f"{scene_id}_vh_clean_2.labels.ply"
    meta_path = scene_dir / f"{scene_id}.txt"
    debug_path = scene_dir / f"{scene_id}.pseudo_debug.json"

    write_xyzrgb_ply(mesh_path, aligned_xyz, rgb)

    with seg_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "sceneId": scene_id,
                "segIndices": [int(x) for x in np.asarray(pseudo["seg_indices"]).tolist()],
            },
            f,
        )

    with agg_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "sceneId": scene_id,
                "appId": "user-ply-pseudo-scannet",
                "segGroups": pseudo["seg_groups"],
            },
            f,
            indent=2,
        )

    write_labels_ply(labels_path, np.asarray(pseudo["semantic_labels"]))

    axis_alignment = np.asarray(pseudo["axis_alignment"], dtype=np.float32)
    with meta_path.open("w", encoding="utf-8") as f:
        f.write(f"sceneId = {scene_id}\n")
        f.write("source = user_ply_pseudo_scannet\n")
        f.write("axisAlignment = " + " ".join(f"{float(x):.8f}" for x in axis_alignment.reshape(-1)) + "\n")

    with debug_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source_input_ply": str(input_ply),
                "note": "Generated from geometry; labels are pseudo labels, not human annotations.",
                "wall_planes": pseudo.get("wall_planes", []),
                "segment_meanings": {
                    "0": "floor_or_unlabeled",
                    "1": "wall",
                    "2": "ceiling_or_top_noise",
                    "10+": "object instances listed in aggregation.json",
                },
            },
            f,
            indent=2,
        )

    if copy_original:
        src = Path(input_ply)
        try:
            shutil.copy2(src, scene_dir / f"{scene_id}.original_input.ply")
        except Exception:
            pass

    return scene_dir


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a raw user .ply room scan into pseudo-ScanNet files and a top-down draggable layout."
    )
    parser.add_argument("--ply", required=True, help="Path to a user room .ply point cloud/mesh")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--scene_id", default=None, help="Optional scene id; defaults to input filename stem")
    parser.add_argument("--up_axis", default="z", choices=["auto", "x", "y", "z"], help="Likely gravity/up axis in the raw PLY. Use y for some ARKit exports; z for ScanNet-like exports.")
    parser.add_argument("--voxel_size", type=float, default=0.05, help="Voxel size for object clustering, in meters")
    parser.add_argument("--plane_distance", type=float, default=0.035, help="RANSAC plane distance threshold, in meters")
    parser.add_argument("--floor_height_threshold", type=float, default=0.08, help="After alignment, points below this z are treated as floor")
    parser.add_argument("--object_min_z", type=float, default=0.12, help="Ignore object candidates below this z after floor alignment")
    parser.add_argument("--cluster_eps", type=float, default=0.18, help="Radius for connected-component clustering, in meters")
    parser.add_argument("--cluster_min_points", type=int, default=12, help="Minimum downsampled points per cluster")
    parser.add_argument("--min_object_points", type=int, default=80, help="Minimum original points per exported object")
    parser.add_argument("--min_object_height", type=float, default=0.12, help="Minimum exported object height, in meters")
    parser.add_argument("--max_object_height", type=float, default=2.4, help="Maximum exported object height, in meters")
    parser.add_argument("--room_resolution_m", type=float, default=0.06, help="Raster resolution used when building the room polygon")
    parser.add_argument("--ransac_iterations", type=int, default=700, help="RANSAC iterations for plane detection")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for RANSAC")
    parser.add_argument("--include_structural", action="store_true", help="Reserved for future use; structural segments are still written as pseudo segments")
    parser.add_argument("--no_png", action="store_true", help="Skip rendering scene_layout.png")
    parser.add_argument("--no_copy_original", action="store_true", help="Do not copy original PLY into pseudo scene folder")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ply_path = Path(args.ply)
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY file not found: {ply_path}")

    scene_id = args.scene_id or ply_path.stem.replace(" ", "_")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layout, pseudo, aligned_xyz, rgb, _, _, _ = build_layout_from_user_ply(
        ply_path=ply_path,
        scene_id=scene_id,
        voxel_size=args.voxel_size,
        plane_distance=args.plane_distance,
        floor_height_threshold=args.floor_height_threshold,
        object_min_z=args.object_min_z,
        cluster_eps=args.cluster_eps,
        cluster_min_points=args.cluster_min_points,
        min_object_points=args.min_object_points,
        min_object_height=args.min_object_height,
        max_object_height=args.max_object_height,
        room_resolution_m=args.room_resolution_m,
        up_axis=args.up_axis,
        ransac_iterations=args.ransac_iterations,
        seed=args.seed,
        include_structural=args.include_structural,
    )

    save_layout_json(layout, out_dir / "scene_layout.json")
    if not args.no_png:
        render_layout_png(layout, out_dir / "scene_layout.png")

    pseudo_scene_dir = write_pseudo_scannet_files(
        out_dir=out_dir,
        scene_id=scene_id,
        input_ply=ply_path,
        aligned_xyz=aligned_xyz,
        rgb=rgb,
        pseudo=pseudo,
        copy_original=not args.no_copy_original,
    )

    print(f"Saved layout JSON to: {out_dir / 'scene_layout.json'}")
    if not args.no_png:
        print(f"Saved layout preview to: {out_dir / 'scene_layout.png'}")
    print(f"Saved pseudo-ScanNet scene folder to: {pseudo_scene_dir}")
    print(f"Objects exported: {len(layout.objects)}")
    print("Open room_editor.html and load scene_layout.json to drag/drop the furniture boxes.")


if __name__ == "__main__":
    main()
