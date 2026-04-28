#!/usr/bin/env python3
"""
High-recall user .ply -> top-down layout exporter.

This is a replacement / companion for export_user_ply_topdown.py. It keeps the
same output contract as the ScanNet exporter:
  - scene_layout.json
  - scene_layout.png
  - pseudo_scannet/scans/<scene_id>/... ScanNet-shaped files

The key difference from the first user-ply implementation is that this version
uses top-down occupancy proposals by default and does NOT remove walls before
proposal generation unless requested. That is usually better for phone/LiDAR
scans, where furniture appears as sparse, disconnected surface fragments.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes, label as cc_label
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
from export_user_ply_topdown import (
    SEMANTIC_ID,
    detect_wall_planes,
    estimate_alignment,
    write_labels_ply,
    write_xyzrgb_ply,
)


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


def rough_label(width: float, depth: float, height: float, z_max: float) -> str:
    long_side = max(width, depth)
    short_side = max(min(width, depth), 1e-6)
    area = width * depth
    aspect = long_side / short_side
    if aspect >= 7.0 and height >= 1.0:
        return "long_thin_surface"
    if area >= 1.35 and height <= 1.35 and z_max <= 1.55:
        return "bed_or_sofa"
    if area >= 0.45 and 0.30 <= height <= 1.35 and z_max <= 1.55:
        return "table_or_desk"
    if area < 0.85 and 0.35 <= height <= 1.60:
        return "chair_or_small_furniture"
    if height >= 1.25 and area >= 0.20:
        return "shelf_or_cabinet"
    if area >= 1.0:
        return "large_furniture"
    if area >= 0.08:
        return "small_furniture"
    return "unknown_object"


def robust_room_polygon(points: np.ndarray, floor_mask: np.ndarray, source: str, resolution: float) -> Tuple[Polygon, str]:
    z = points[:, 2]
    xy = points[:, :2]
    source = source.lower()
    if source == "floor":
        return room_polygon_from_floor_points(xy[floor_mask], xy, resolution), "floor"
    if source == "lower_slice":
        cutoff = max(float(np.percentile(z, 20)), 0.30)
        return room_polygon_from_floor_points(xy[z <= cutoff], xy, resolution), f"lower_slice_z<={cutoff:.3f}"
    # full trimmed hull is most forgiving when floor points are sparse or hidden
    lo = np.percentile(xy, 0.5, axis=0)
    hi = np.percentile(xy, 99.5, axis=0)
    keep = np.all((xy >= lo) & (xy <= hi), axis=1)
    trimmed = xy[keep]
    if len(trimmed) < 20:
        trimmed = xy
    poly = MultiPoint(trimmed).convex_hull
    if poly.is_empty or not isinstance(poly, Polygon):
        poly = MultiPoint(xy).convex_hull
    return poly.simplify(resolution, preserve_topology=True), "full_trimmed_convex_hull"


def occupancy_components(
    points: np.ndarray,
    candidate_mask: np.ndarray,
    resolution: float,
    close_iters: int,
    dilate_iters: int,
    min_component_cells: int,
) -> Tuple[List[np.ndarray], Dict[str, object]]:
    candidate_idx = np.where(candidate_mask)[0]
    if len(candidate_idx) == 0:
        return [], {"candidate_points": 0, "components_raw": 0, "components_kept": 0}
    xy = points[candidate_idx, :2]
    mn = xy.min(axis=0) - 2 * resolution
    mx = xy.max(axis=0) + 2 * resolution
    wh = np.maximum(np.ceil((mx - mn) / resolution).astype(int) + 1, 3)
    if int(wh[0] * wh[1]) > 12_000_000:
        raise RuntimeError(f"Occupancy grid too large ({wh[0]} x {wh[1]}). Increase --occupancy_resolution.")
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
    labeled, ncomp = cc_label(grid, structure=structure)
    point_component = labeled[ij[:, 1], ij[:, 0]]
    comps: List[np.ndarray] = []
    for cid in range(1, ncomp + 1):
        cell_count = int((labeled == cid).sum())
        if cell_count < min_component_cells:
            continue
        verts = candidate_idx[point_component == cid]
        if len(verts):
            comps.append(verts.astype(np.int64))
    return comps, {
        "candidate_points": int(len(candidate_idx)),
        "grid_width": int(wh[0]),
        "grid_height": int(wh[1]),
        "resolution": float(resolution),
        "components_raw": int(ncomp),
        "components_kept_by_cell_count": int(len(comps)),
    }


def trim_component(points: np.ndarray) -> np.ndarray:
    if len(points) < 30:
        return points
    lo = np.percentile(points, 0.5, axis=0)
    hi = np.percentile(points, 99.5, axis=0)
    keep = np.all((points >= lo) & (points <= hi), axis=1)
    if keep.sum() < max(15, 0.4 * len(points)):
        return points
    return points[keep]


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

    wall_base = (z > args.floor_height_threshold) & (z < max(float(np.percentile(z, 98)), 1.2))
    wall_planes, wall_mask = detect_wall_planes(
        aligned,
        base_mask=wall_base,
        plane_distance=max(args.plane_distance * 1.5, 0.04),
        ransac_iterations=args.ransac_iterations,
        rng=rng,
    )
    if args.wall_removal == "hard":
        wall_remove = wall_mask
    elif args.wall_removal == "soft":
        wall_remove = wall_mask & (z > 1.15)
    else:
        wall_remove = np.zeros(len(aligned), dtype=bool)

    object_max_z = float(args.object_max_z if args.object_max_z > 0 else np.percentile(z, 99.5))
    candidate_mask = (z >= args.object_min_z) & (z <= object_max_z) & (~wall_remove)

    room_poly, room_source_used = robust_room_polygon(aligned, floor_mask, args.room_source, args.room_resolution_m)

    comps, proposal_debug = occupancy_components(
        aligned,
        candidate_mask,
        resolution=args.occupancy_resolution,
        close_iters=args.xy_close_iters,
        dilate_iters=args.xy_dilate_iters,
        min_component_cells=args.min_component_cells,
    )

    seg_indices = np.zeros(len(aligned), dtype=np.int32)
    semantic_labels = np.full(len(aligned), SEMANTIC_ID["unlabeled"], dtype=np.int32)
    semantic_labels[floor_mask] = SEMANTIC_ID["floor"]
    semantic_labels[wall_mask] = SEMANTIC_ID["wall"]
    seg_indices[floor_mask] = 0
    seg_indices[wall_mask] = 1

    objects: List[Object2D] = []
    seg_groups: List[dict] = []
    comp_debug: List[dict] = []
    next_id = 1
    next_seg = 10

    for i, verts in enumerate(comps, start=1):
        if len(verts) < args.min_object_points:
            comp_debug.append({"component": i, "status": "rejected_min_points", "points": int(len(verts))})
            continue
        pts = trim_component(aligned[verts])
        if len(pts) < 5:
            pts = aligned[verts]
        xy = pts[:, :2]
        center, width, depth, theta, _ = oriented_bbox_2d(xy)
        z_min = float(np.percentile(pts[:, 2], 1.0))
        z_max = float(np.percentile(pts[:, 2], 99.0))
        height = max(0.0, z_max - z_min)
        area = float(width * depth)
        reason = None
        if height < args.min_object_height:
            reason = "rejected_min_height"
        elif height > args.max_object_height:
            reason = "rejected_max_height"
        elif max(width, depth) < 0.05 or area < args.min_box_area:
            reason = "rejected_min_area"
        hull = MultiPoint(xy).convex_hull
        outside_distance = float(room_poly.distance(hull)) if not hull.is_empty else 0.0
        if args.filter_outside_room and outside_distance > 0.40:
            reason = "rejected_outside_room"
        if reason:
            comp_debug.append({
                "component": i,
                "status": reason,
                "points": int(len(verts)),
                "width": float(width),
                "depth": float(depth),
                "height": float(height),
                "area": float(area),
                "outside_distance": outside_distance,
            })
            continue

        width_p = max(float(width + 2 * args.box_padding), 0.05)
        depth_p = max(float(depth + 2 * args.box_padding), 0.05)
        corners = bbox_corners(float(center[0]), float(center[1]), width_p, depth_p, float(theta))
        label = rough_label(width_p, depth_p, height, z_max)
        sem_id = SEMANTIC_ID.get(label, SEMANTIC_ID["unknown_object"])
        seg_indices[verts] = next_seg
        semantic_labels[verts] = sem_id
        center_3d = pts.mean(axis=0)
        size_3d = pts.max(axis=0) - pts.min(axis=0)
        objects.append(Object2D(
            id=next_id,
            object_id=next_id,
            label=label,
            raw_label=label,
            cx=float(center[0]),
            cy=float(center[1]),
            width=width_p,
            depth=depth_p,
            theta=float(theta),
            z_min=z_min,
            z_max=z_max,
            height=float(height),
            point_count=int(len(verts)),
            footprint=[[float(x), float(y)] for x, y in corners.tolist()],
            bbox3d_center=[float(v) for v in center_3d.tolist()],
            bbox3d_size=[float(v) for v in size_3d.tolist()],
        ))
        seg_groups.append({"objectId": int(next_id), "id": int(next_id), "label": label, "segments": [int(next_seg)]})
        comp_debug.append({
            "component": i,
            "status": "kept",
            "object_id": int(next_id),
            "label": label,
            "points": int(len(verts)),
            "width": width_p,
            "depth": depth_p,
            "height": float(height),
            "outside_distance": outside_distance,
        })
        next_id += 1
        next_seg += 1

    objects.sort(key=lambda o: (-(o.width * o.depth), o.label.lower(), o.id))
    scene_id = args.scene_id or ply.stem.replace(" ", "_")
    metadata = {
        "source": "user_ply_geometry_pseudo_scannet_v2",
        "input_ply": str(ply),
        "axis_alignment_estimated": not args.assume_aligned,
        "axis_alignment_matrix": [float(x) for x in axis_alignment.reshape(-1).tolist()],
        "alignment": align_meta,
        "proposal_mode": "xy_occupancy_high_recall",
        "wall_removal": args.wall_removal,
        "floor_height_threshold": float(args.floor_height_threshold),
        "object_min_z": float(args.object_min_z),
        "object_max_z": float(object_max_z),
        "occupancy_resolution": float(args.occupancy_resolution),
        "xy_close_iters": int(args.xy_close_iters),
        "xy_dilate_iters": int(args.xy_dilate_iters),
        "min_component_cells": int(args.min_component_cells),
        "min_object_points": int(args.min_object_points),
        "min_object_height": float(args.min_object_height),
        "max_object_height": float(args.max_object_height),
        "min_box_area": float(args.min_box_area),
        "box_padding": float(args.box_padding),
        "room_source_requested": args.room_source,
        "room_source_used": room_source_used,
        "raw_vertex_count": int(len(raw_xyz)),
        "floor_vertex_count": int(floor_mask.sum()),
        "detected_wall_vertex_count": int(wall_mask.sum()),
        "wall_removed_candidate_vertex_count": int(wall_remove.sum()),
        "object_candidate_vertex_count": int(candidate_mask.sum()),
        "candidate_vertex_ratio": float(candidate_mask.mean()),
        "component_count_before_filters": int(len(comps)),
        "object_count": int(len(objects)),
        "filter_outside_room": bool(args.filter_outside_room),
        "labeling_method": "geometry-only high-recall proposals; replace with ML instance segmentation for robust labels",
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
        "candidate_mask": candidate_mask,
        "seg_indices": seg_indices,
        "semantic_labels": semantic_labels,
        "seg_groups": seg_groups,
        "wall_planes": [{"normal": [float(x) for x in n.tolist()], "d": float(d)} for n, d in wall_planes],
    }
    debug = {"proposal_debug": proposal_debug, "components": comp_debug, "wall_planes": pseudo["wall_planes"]}
    return layout, pseudo, aligned, rgb, debug


def write_pseudo_scannet_files(out_dir: str | Path, scene_id: str, input_ply: str | Path, aligned: np.ndarray, rgb: Optional[np.ndarray], pseudo: Dict[str, object], copy_original: bool = True) -> Path:
    scene_dir = Path(out_dir) / "pseudo_scannet" / "scans" / scene_id
    scene_dir.mkdir(parents=True, exist_ok=True)
    write_xyzrgb_ply(scene_dir / f"{scene_id}_vh_clean_2.ply", aligned, rgb)
    with (scene_dir / f"{scene_id}_vh_clean_2.0.010000.segs.json").open("w", encoding="utf-8") as f:
        json.dump({"sceneId": scene_id, "segIndices": [int(x) for x in np.asarray(pseudo["seg_indices"]).tolist()]}, f)
    with (scene_dir / f"{scene_id}.aggregation.json").open("w", encoding="utf-8") as f:
        json.dump({"sceneId": scene_id, "appId": "user-ply-pseudo-scannet-v2", "segGroups": pseudo["seg_groups"]}, f, indent=2)
    write_labels_ply(scene_dir / f"{scene_id}_vh_clean_2.labels.ply", np.asarray(pseudo["semantic_labels"]))
    axis = np.asarray(pseudo["axis_alignment"], dtype=np.float32)
    with (scene_dir / f"{scene_id}.txt").open("w", encoding="utf-8") as f:
        f.write(f"sceneId = {scene_id}\n")
        f.write("source = user_ply_pseudo_scannet_v2\n")
        f.write("axisAlignment = " + " ".join(f"{float(x):.8f}" for x in axis.reshape(-1)) + "\n")
    with (scene_dir / f"{scene_id}.pseudo_debug.json").open("w", encoding="utf-8") as f:
        json.dump({
            "source_input_ply": str(input_ply),
            "note": "Generated from geometry; labels are pseudo labels, not human annotations.",
            "wall_planes": pseudo.get("wall_planes", []),
            "segment_meanings": {"0": "floor_or_unlabeled", "1": "wall", "10+": "object proposal segments"},
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
    idx = rng.choice(n, size=min(n, 120000), replace=False) if n > 120000 else np.arange(n)
    pts = aligned[idx]
    floor = np.asarray(pseudo["floor_mask"])[idx]
    wall = np.asarray(pseudo["wall_mask"])[idx]
    cand = np.asarray(pseudo["candidate_mask"])[idx]
    plt.figure(figsize=(10, 10))
    plt.scatter(pts[:, 0], pts[:, 1], s=0.2, alpha=0.08, label="all sampled")
    if floor.any():
        p = pts[floor]
        plt.scatter(p[:, 0], p[:, 1], s=0.35, alpha=0.20, label="floor")
    if wall.any():
        p = pts[wall]
        plt.scatter(p[:, 0], p[:, 1], s=0.35, alpha=0.18, label="detected walls")
    if cand.any():
        p = pts[cand]
        plt.scatter(p[:, 0], p[:, 1], s=0.45, alpha=0.35, label="object candidates")
    room = np.asarray(layout.room_polygon, dtype=float)
    if len(room) >= 3:
        closed = np.vstack([room, room[0]])
        plt.plot(closed[:, 0], closed[:, 1], linewidth=2.0, label="room polygon")
    for obj in layout.objects:
        corners = np.asarray(obj.footprint, dtype=float)
        closed = np.vstack([corners, corners[0]])
        plt.plot(closed[:, 0], closed[:, 1], linewidth=1.5)
        plt.text(obj.cx, obj.cy, f"{obj.id}:{obj.label}", fontsize=7)
    plt.axis("equal")
    plt.legend(loc="best", markerscale=8)
    plt.title("Debug layers: candidates and exported boxes")
    plt.tight_layout()
    plt.savefig(out_dir / "debug_layers.png", dpi=180)
    plt.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="High-recall raw user .ply -> top-down draggable layout exporter")
    p.add_argument("--ply", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--scene_id", default=None)
    p.add_argument("--up_axis", default="z", choices=["auto", "x", "y", "z"])
    p.add_argument("--assume_aligned", action="store_true")
    p.add_argument("--plane_distance", type=float, default=0.04)
    p.add_argument("--ransac_iterations", type=int, default=800)
    p.add_argument("--floor_height_threshold", type=float, default=0.06)
    p.add_argument("--wall_removal", default="hard", choices=["none", "soft", "hard"])
    p.add_argument("--object_min_z", type=float, default=0.06)
    p.add_argument("--object_max_z", type=float, default=2.2)
    p.add_argument("--occupancy_resolution", type=float, default=0.08)
    p.add_argument("--xy_close_iters", type=int, default=0)
    p.add_argument("--xy_dilate_iters", type=int, default=0)
    p.add_argument("--min_component_cells", type=int, default=3)
    p.add_argument("--min_object_points", type=int, default=25)
    p.add_argument("--min_object_height", type=float, default=0.05)
    p.add_argument("--max_object_height", type=float, default=2.6)
    p.add_argument("--min_box_area", type=float, default=0.015)
    p.add_argument("--box_padding", type=float, default=0.12)
    p.add_argument("--room_source", default="full", choices=["full", "floor", "lower_slice"])
    p.add_argument("--room_resolution_m", type=float, default=0.08)
    p.add_argument("--filter_outside_room", action="store_true")
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
    print(f"Saved pseudo-ScanNet scene folder to: {pseudo_dir}")
    print(f"Objects exported: {len(layout.objects)}")


if __name__ == "__main__":
    main()
