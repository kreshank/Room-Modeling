#!/usr/bin/env python3
"""
Export a ScanNet scene into a labeled top-down room layout JSON + PNG.

This script is designed for ScanNet scenes that already include:
- <scene>_vh_clean_2.ply
- <scene>.aggregation.json
- <scene>_vh_clean_2.0.010000.segs.json
- <scene>.txt
- <scene>_vh_clean_2.labels.ply
- scannetv2-labels.combined.tsv

Output:
- scene_layout.json : room polygon + draggable furniture rectangles
- scene_layout.png  : preview image

Why this script exists:
It converts ScanNet's raw 3D mesh + annotations into a clean 2D scene graph
that is much easier to manipulate in a room-layout editor.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import trimesh
from scipy.ndimage import binary_closing, binary_dilation, binary_fill_holes
from shapely.geometry import MultiPoint, Polygon, box
from shapely.ops import unary_union


PLY_NUMPY_DTYPES = {
    "char": np.int8,
    "uchar": np.uint8,
    "short": np.int16,
    "ushort": np.uint16,
    "int": np.int32,
    "uint": np.uint32,
    "float": np.float32,
    "double": np.float64,
}

MOVABLE_EXCLUDE_LABELS = {
    "wall",
    "floor",
    "ceiling",
    "window",
    "door",
    "curtain",
    "shower curtain",
    "mirror",
    "otherstructure",
    "otherprop",
    "otherfurniture",  # comment this out if you want more instances in the editor
}


@dataclass
class Object2D:
    id: int
    object_id: int
    label: str
    raw_label: str
    cx: float
    cy: float
    width: float
    depth: float
    theta: float
    z_min: float
    z_max: float
    height: float
    point_count: int
    footprint: List[List[float]]
    bbox3d_center: List[float]
    bbox3d_size: List[float]


@dataclass
class SceneLayout:
    scene_id: str
    units: str
    room_polygon: List[List[float]]
    room_bbox: Dict[str, float]
    objects: List[Object2D]
    metadata: Dict[str, object]


# ----------------------------
# Low-level parsing utilities
# ----------------------------

def read_vertex_table_from_ply(path: str | Path, wanted_properties: Optional[Sequence[str]] = None) -> Dict[str, np.ndarray]:
    """Read vertex properties from an ASCII or binary little-endian PLY.

    ScanNet's relevant PLY files store per-vertex data. We only need the vertex
    table, not the face data.
    """
    path = Path(path)
    with path.open("rb") as f:
        fmt = None
        vertex_count = None
        in_vertex_element = False
        vertex_props: List[Tuple[str, str]] = []

        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected end of file while reading PLY header: {path}")
            line_s = line.decode("ascii", errors="ignore").strip()
            if line_s.startswith("format "):
                fmt = line_s.split()[1]
            elif line_s.startswith("element "):
                parts = line_s.split()
                in_vertex_element = parts[1] == "vertex"
                if in_vertex_element:
                    vertex_count = int(parts[2])
            elif line_s.startswith("property ") and in_vertex_element:
                parts = line_s.split()
                if parts[1] == "list":
                    raise ValueError(f"Unsupported list property in vertex element for {path}")
                vertex_props.append((parts[1], parts[2]))
            elif line_s == "end_header":
                break

        if fmt is None or vertex_count is None:
            raise ValueError(f"Malformed PLY header in {path}")

        if wanted_properties is None:
            wanted_properties = [name for _, name in vertex_props]
        wanted_properties = set(wanted_properties)

        if fmt == "binary_little_endian":
            dtype = np.dtype([(name, PLY_NUMPY_DTYPES[ptype]) for ptype, name in vertex_props])
            arr = np.fromfile(f, dtype=dtype, count=vertex_count)
            return {name: np.asarray(arr[name]) for _, name in vertex_props if name in wanted_properties}

        if fmt == "ascii":
            cols = {name: [] for _, name in vertex_props if name in wanted_properties}
            prop_names = [name for _, name in vertex_props]
            converters = [PLY_NUMPY_DTYPES[ptype] for ptype, _ in vertex_props]
            for _ in range(vertex_count):
                parts = f.readline().decode("ascii", errors="ignore").strip().split()
                if len(parts) != len(prop_names):
                    raise ValueError(f"Unexpected ASCII vertex row length in {path}")
                for name, conv, token in zip(prop_names, converters, parts):
                    if name in cols:
                        cols[name].append(conv(token))
            return {k: np.asarray(v) for k, v in cols.items()}

        raise ValueError(f"Unsupported PLY format '{fmt}' in {path}")


def load_mesh_vertices(mesh_path: str | Path) -> np.ndarray:
    props = read_vertex_table_from_ply(mesh_path, ["x", "y", "z", "red", "green", "blue"])
    xyz = np.column_stack([props["x"], props["y"], props["z"]]).astype(np.float32)
    return xyz


def load_semantic_labels(labels_ply_path: str | Path) -> np.ndarray:
    props = read_vertex_table_from_ply(labels_ply_path, ["label"])
    if "label" not in props:
        raise ValueError(f"No 'label' vertex property found in {labels_ply_path}")
    return props["label"].astype(np.int32)


def load_axis_alignment(meta_path: str | Path) -> np.ndarray:
    meta_path = Path(meta_path)
    with meta_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "axisAlignment" in line:
                values = [float(x) for x in line.split("=")[1].strip().split()]
                if len(values) != 16:
                    raise ValueError(f"axisAlignment in {meta_path} does not have 16 values")
                return np.asarray(values, dtype=np.float32).reshape(4, 4)
    return np.eye(4, dtype=np.float32)


def apply_transform(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=points.dtype)
    homo = np.hstack([points, ones])
    transformed = homo @ transform.T
    return transformed[:, :3]


def load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_label_map(label_map_path: str | Path) -> List[dict]:
    with open(label_map_path, "r", encoding="utf-8-sig", errors="ignore") as f:
        reader = csv.DictReader(f, delimiter="\t")
        return list(reader)


# ----------------------------
# Geometry helpers
# ----------------------------

def angle_wrap(theta: float) -> float:
    while theta <= -math.pi:
        theta += 2 * math.pi
    while theta > math.pi:
        theta -= 2 * math.pi
    return theta


def oriented_bbox_2d(points_xy: np.ndarray) -> Tuple[np.ndarray, float, float, float, np.ndarray]:
    """Return center, width, depth, theta, corners for a 2D oriented box using PCA."""
    if len(points_xy) < 3:
        min_xy = points_xy.min(axis=0)
        max_xy = points_xy.max(axis=0)
        center = (min_xy + max_xy) / 2.0
        size = np.maximum(max_xy - min_xy, 1e-3)
        theta = 0.0
        corners = np.array([
            [min_xy[0], min_xy[1]],
            [max_xy[0], min_xy[1]],
            [max_xy[0], max_xy[1]],
            [min_xy[0], max_xy[1]],
        ])
        return center, float(size[0]), float(size[1]), theta, corners

    centered = points_xy - points_xy.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, np.argmax(eigvals)]
    theta = math.atan2(axis[1], axis[0])

    c, s = math.cos(-theta), math.sin(-theta)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32)
    pts_rot = centered @ rot.T

    min_xy = pts_rot.min(axis=0)
    max_xy = pts_rot.max(axis=0)
    width, depth = (max_xy - min_xy).tolist()
    center_rot = (min_xy + max_xy) / 2.0
    center_world = center_rot @ rot + points_xy.mean(axis=0)

    corners_rot = np.array([
        [min_xy[0], min_xy[1]],
        [max_xy[0], min_xy[1]],
        [max_xy[0], max_xy[1]],
        [min_xy[0], max_xy[1]],
    ], dtype=np.float32)
    corners_world = corners_rot @ rot + points_xy.mean(axis=0)
    return center_world, float(width), float(depth), angle_wrap(theta), corners_world


def room_polygon_from_floor_points(
    floor_xy: np.ndarray,
    fallback_xy: np.ndarray,
    resolution: float = 0.05,
) -> Polygon:
    """Build a room polygon from floor points using a raster occupancy approach.

    This is intentionally robust and practical. It usually works better than a
    bare convex hull for messy floor scans and is easy to simplify afterwards.
    """
    if floor_xy is None or len(floor_xy) < 20:
        return MultiPoint(fallback_xy).convex_hull

    margin = 2
    mn = floor_xy.min(axis=0) - resolution * margin
    ij = np.floor((floor_xy - mn) / resolution).astype(np.int32)
    ij[:, 0] -= ij[:, 0].min()
    ij[:, 1] -= ij[:, 1].min()

    width = int(ij[:, 0].max()) + 1
    height = int(ij[:, 1].max()) + 1
    grid = np.zeros((height, width), dtype=bool)
    grid[ij[:, 1], ij[:, 0]] = True

    structure = np.ones((3, 3), dtype=bool)
    grid = binary_closing(grid, structure=structure)
    grid = binary_fill_holes(grid)
    grid = binary_dilation(grid, structure=structure)

    ys, xs = np.nonzero(grid)
    if len(xs) == 0:
        return MultiPoint(fallback_xy).convex_hull

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

    hull = MultiPoint(floor_xy).convex_hull
    if poly.is_empty or poly.area < 0.2 * max(hull.area, 1e-6):
        poly = hull

    poly = poly.simplify(resolution, preserve_topology=True)
    return poly


def polygon_to_coords(poly: Polygon) -> List[List[float]]:
    coords = np.asarray(poly.exterior.coords)
    return [[float(x), float(y)] for x, y in coords[:-1]]


def polygon_bbox(poly: Polygon) -> Dict[str, float]:
    minx, miny, maxx, maxy = poly.bounds
    return {
        "min_x": float(minx),
        "min_y": float(miny),
        "max_x": float(maxx),
        "max_y": float(maxy),
        "width": float(maxx - minx),
        "height": float(maxy - miny),
    }


# ----------------------------
# Scene conversion
# ----------------------------

def infer_floor_label_ids(label_rows: List[dict]) -> set[int]:
    ids: set[int] = set()
    for row in label_rows:
        raw_cat = (row.get("raw_category") or "").strip().lower()
        nyu40class = (row.get("nyu40class") or "").strip().lower()
        if raw_cat == "floor" or nyu40class == "floor":
            for key in ("id", "nyu40id"):
                value = (row.get(key) or "").strip()
                if value and value not in {"-", "ignore"}:
                    try:
                        ids.add(int(value))
                    except ValueError:
                        pass
    if not ids:
        ids.add(2)  # common NYU40 floor id fallback
    return ids


def build_segment_to_vertices(seg_indices: Sequence[int]) -> Dict[int, np.ndarray]:
    seg_to_vertices: Dict[int, List[int]] = {}
    for vertex_idx, seg_id in enumerate(seg_indices):
        seg_to_vertices.setdefault(int(seg_id), []).append(vertex_idx)
    return {k: np.asarray(v, dtype=np.int32) for k, v in seg_to_vertices.items()}


def object_vertices_from_group(seg_group: dict, seg_to_vertices: Dict[int, np.ndarray]) -> np.ndarray:
    vertex_lists = [seg_to_vertices.get(int(seg_id), np.empty((0,), dtype=np.int32)) for seg_id in seg_group.get("segments", [])]
    if not vertex_lists:
        return np.empty((0,), dtype=np.int32)
    return np.unique(np.concatenate(vertex_lists))


def find_scene_files(scene_dir: str | Path) -> Dict[str, Path]:
    scene_dir = Path(scene_dir)
    if not scene_dir.exists():
        raise FileNotFoundError(f"Scene directory not found: {scene_dir}")

    scene_id = scene_dir.name
    files = {
        "scene_id": scene_id,
        "mesh": scene_dir / f"{scene_id}_vh_clean_2.ply",
        "agg": scene_dir / f"{scene_id}.aggregation.json",
        "seg": scene_dir / f"{scene_id}_vh_clean_2.0.010000.segs.json",
        "meta": scene_dir / f"{scene_id}.txt",
        "labels": scene_dir / f"{scene_id}_vh_clean_2.labels.ply",
    }
    missing = [str(p) for k, p in files.items() if k != "scene_id" and not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required scene files:\n" + "\n".join(missing))
    return files


def convert_scene(
    scene_dir: str | Path,
    label_map_path: str | Path,
    include_non_movable: bool = False,
    min_points_per_object: int = 20,
    room_resolution_m: float = 0.05,
) -> SceneLayout:
    files = find_scene_files(scene_dir)
    scene_id = files["scene_id"]

    mesh_xyz = load_mesh_vertices(files["mesh"])
    semantic_labels = load_semantic_labels(files["labels"])
    axis_alignment = load_axis_alignment(files["meta"])
    mesh_xyz = apply_transform(mesh_xyz, axis_alignment)

    agg = load_json(files["agg"])
    seg = load_json(files["seg"])
    seg_indices = seg["segIndices"]
    if len(seg_indices) != len(mesh_xyz):
        raise ValueError(
            f"Vertex count mismatch in scene {scene_id}: mesh has {len(mesh_xyz)} vertices but segIndices has {len(seg_indices)}"
        )

    label_rows = load_label_map(label_map_path)
    floor_label_ids = infer_floor_label_ids(label_rows)
    floor_mask = np.isin(semantic_labels, list(floor_label_ids))
    floor_xy = mesh_xyz[floor_mask, :2]

    room_poly = room_polygon_from_floor_points(floor_xy, mesh_xyz[:, :2], resolution=room_resolution_m)
    seg_to_vertices = build_segment_to_vertices(seg_indices)

    objects: List[Object2D] = []
    next_id = 1
    for seg_group in agg.get("segGroups", []):
        raw_label = str(seg_group.get("label", "object")).strip()
        label = raw_label.lower()
        if not include_non_movable and label in MOVABLE_EXCLUDE_LABELS:
            continue

        vertex_ids = object_vertices_from_group(seg_group, seg_to_vertices)
        if len(vertex_ids) < min_points_per_object:
            continue

        obj_points = mesh_xyz[vertex_ids]
        xy = obj_points[:, :2]
        center_xy, width, depth, theta, corners = oriented_bbox_2d(xy)
        z_min = float(obj_points[:, 2].min())
        z_max = float(obj_points[:, 2].max())
        center_3d = obj_points.mean(axis=0)
        size_3d = obj_points.max(axis=0) - obj_points.min(axis=0)

        # Clip objects that are far outside the detected room footprint.
        # This mostly protects against weird scan fragments.
        if not room_poly.buffer(0.15).contains(MultiPoint(xy).convex_hull.centroid):
            # Keep tall objects close to the border, but reject obvious outliers.
            if room_poly.distance(MultiPoint(xy).convex_hull) > 0.40:
                continue

        objects.append(
            Object2D(
                id=next_id,
                object_id=int(seg_group.get("objectId", next_id)),
                label=raw_label,
                raw_label=raw_label,
                cx=float(center_xy[0]),
                cy=float(center_xy[1]),
                width=float(max(width, 0.05)),
                depth=float(max(depth, 0.05)),
                theta=float(theta),
                z_min=z_min,
                z_max=z_max,
                height=float(max(z_max - z_min, 0.0)),
                point_count=int(len(vertex_ids)),
                footprint=[[float(x), float(y)] for x, y in corners.tolist()],
                bbox3d_center=[float(v) for v in center_3d.tolist()],
                bbox3d_size=[float(v) for v in size_3d.tolist()],
            )
        )
        next_id += 1

    objects.sort(key=lambda o: (o.label.lower(), o.id))
    layout = SceneLayout(
        scene_id=scene_id,
        units="meters",
        room_polygon=polygon_to_coords(room_poly),
        room_bbox=polygon_bbox(room_poly),
        objects=objects,
        metadata={
            "source": "ScanNet annotated scene",
            "mesh_file": Path(files["mesh"]).name,
            "aggregation_file": Path(files["agg"]).name,
            "segmentation_file": Path(files["seg"]).name,
            "meta_file": Path(files["meta"]).name,
            "labels_file": Path(files["labels"]).name,
            "floor_label_ids_used": sorted(int(x) for x in floor_label_ids),
            "axis_alignment_applied": True,
            "room_resolution_m": room_resolution_m,
            "object_count": len(objects),
            "excluded_labels": sorted(MOVABLE_EXCLUDE_LABELS) if not include_non_movable else [],
            "min_points_per_object": min_points_per_object,
        },
    )
    return layout


# ----------------------------
# Output helpers
# ----------------------------

def save_layout_json(layout: SceneLayout, out_path: str | Path) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(layout)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def render_layout_png(layout: SceneLayout, out_path: str | Path, show_labels: bool = True) -> None:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 10))
    room = np.asarray(layout.room_polygon)
    if len(room) >= 3:
        ax.fill(room[:, 0], room[:, 1], alpha=0.08, color="black")
        ax.plot(np.r_[room[:, 0], room[0, 0]], np.r_[room[:, 1], room[0, 1]], linewidth=2)

    for obj in layout.objects:
        c = np.array([obj.cx, obj.cy], dtype=np.float32)
        w, d, theta = obj.width, obj.depth, obj.theta
        corners = np.array([
            [-w / 2, -d / 2],
            [ w / 2, -d / 2],
            [ w / 2,  d / 2],
            [-w / 2,  d / 2],
        ], dtype=np.float32)
        rot = np.array([
            [math.cos(theta), -math.sin(theta)],
            [math.sin(theta),  math.cos(theta)],
        ], dtype=np.float32)
        corners = corners @ rot.T + c
        poly = patches.Polygon(corners, closed=True, fill=False, linewidth=1.5)
        ax.add_patch(poly)
        ax.plot([c[0], c[0] + (w / 2) * math.cos(theta)], [c[1], c[1] + (w / 2) * math.sin(theta)], linewidth=1)
        if show_labels:
            ax.text(c[0], c[1], obj.label, fontsize=8, ha="center", va="center")

    bbox = layout.room_bbox
    margin = max(bbox["width"], bbox["height"], 1.0) * 0.05
    ax.set_xlim(bbox["min_x"] - margin, bbox["max_x"] + margin)
    ax.set_ylim(bbox["min_y"] - margin, bbox["max_y"] + margin)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{layout.scene_id} top-down layout")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a ScanNet scene into a top-down draggable layout JSON.")
    parser.add_argument("--scene_dir", required=True, help="Path to one scene directory, e.g. data/scannet/scans/scene0000_00")
    parser.add_argument("--label_map", required=True, help="Path to scannetv2-labels.combined.tsv")
    parser.add_argument("--out_dir", required=True, help="Output directory for scene_layout.json and scene_layout.png")
    parser.add_argument("--include_non_movable", action="store_true", help="Keep walls/floors/windows/etc. as objects too")
    parser.add_argument("--min_points_per_object", type=int, default=20, help="Discard tiny noisy instances")
    parser.add_argument("--room_resolution_m", type=float, default=0.05, help="Raster resolution used when building the room polygon")
    parser.add_argument("--no_png", action="store_true", help="Skip rendering the PNG preview")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layout = convert_scene(
        scene_dir=args.scene_dir,
        label_map_path=args.label_map,
        include_non_movable=args.include_non_movable,
        min_points_per_object=args.min_points_per_object,
        room_resolution_m=args.room_resolution_m,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "scene_layout.json"
    png_path = out_dir / "scene_layout.png"

    save_layout_json(layout, json_path)
    if not args.no_png:
        render_layout_png(layout, png_path)

    print(f"Saved layout JSON to: {json_path}")
    if not args.no_png:
        print(f"Saved layout preview to: {png_path}")
    print(f"Objects exported: {len(layout.objects)}")


if __name__ == "__main__":
    main()
