"""Convert a `scene_graph.json` (from the `graph/` pipeline) into a PyG
`HeteroData` object.

Edge types from the scene graph are **collapsed** into a small set of
`(src_kind, relation_bucket, dst_kind)` triplets (spatial / wall / room /
zone / …). The original edge type string is preserved as a one-hot in
`edge_attr`, so the attention layers still see ``near`` vs ``faces`` etc.

`principle_*` edges are filtered out — they are rule-engine labels.

Undirected connectivity uses PyG's ``ToUndirected`` so destination-only nodes
(walls, rooms, zones) receive gradients without hand-written reverse-edge
suffixes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected

from .labels import (
    EDGE_KIND_VOCAB_SIZE,
    KIND_VOCAB,
    edge_kind_to_index,
    label_to_index,
)


# Per-node continuous feature schema:
#   cx, cy, z, yaw_sin, yaw_cos, width, depth, height
NODE_FEAT_DIM: int = 8

# Per-edge continuous feature schema. Missing keys are zero-padded.
EDGE_FEAT_KEYS: tuple[str, ...] = (
    "distance_m",
    "angle_diff_deg",
    "angle_error_deg",
    "clearance_m",
    "yaw_diff_deg",
    "coverage",
    "overlap_m2",
    "length_m",
)
CONT_EDGE_DIM: int = len(EDGE_FEAT_KEYS)

# Full edge_attr = [continuous …] ++ one-hot(original edge type name)
EDGE_FEAT_DIM: int = CONT_EDGE_DIM + EDGE_KIND_VOCAB_SIZE

EXCLUDED_EDGE_PREFIXES: tuple[str, ...] = ("principle_",)


def load_scene_graph(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _node_feature_vector(node: dict[str, Any]) -> list[float]:
    geom = node.get("geometry", {}) or {}
    cx = float(geom.get("cx", 0.0))
    cy = float(geom.get("cy", 0.0))
    centroid = geom.get("centroid_xy")
    if isinstance(centroid, (list, tuple)) and len(centroid) >= 2:
        cx, cy = float(centroid[0]), float(centroid[1])
    z = float(geom.get("z", 0.0))
    yaw = float(geom.get("yaw_rad", 0.0))
    width = float(geom.get("width", 0.0))
    depth = float(geom.get("depth", 0.0))
    height = float(geom.get("height", 0.0))
    if width == 0.0 and depth == 0.0 and "area_m2" in geom:
        side = math.sqrt(max(0.0, float(geom["area_m2"])))
        width = depth = side
    return [cx, cy, z, math.sin(yaw), math.cos(yaw), width, depth, height]


def _continuous_edge_feats(edge: dict[str, Any]) -> list[float]:
    out: list[float] = []
    for key in EDGE_FEAT_KEYS:
        value = edge.get(key, 0.0)
        try:
            out.append(float(value))
        except (TypeError, ValueError):
            out.append(0.0)
    return out


def _edge_attr_row(edge_type: str, edge: dict[str, Any]) -> list[float]:
    cont = _continuous_edge_feats(edge)
    ek = edge_kind_to_index(edge_type)
    one_hot = [0.0] * EDGE_KIND_VOCAB_SIZE
    one_hot[ek] = 1.0
    return cont + one_hot


def _is_excluded_edge_type(edge_type: str) -> bool:
    return any(edge_type.startswith(p) for p in EXCLUDED_EDGE_PREFIXES)


def _relation_bucket(s_type: str, d_type: str, edge_type: str) -> str | None:
    """Map many scene-graph edge type strings onto a small convolution bucket."""

    if s_type == "object" and d_type == "object":
        return "spatial"
    if s_type == "object" and d_type == "wall":
        return "to_wall"
    if s_type in ("door", "window") and d_type == "wall":
        return "aperture_wall"
    if s_type in ("object", "door", "window") and d_type == "room":
        return "to_room"
    if s_type == "object" and d_type == "zone":
        return "to_zone"
    if s_type == "door" and d_type == "zone":
        return "door_to_zone"
    if s_type == "door" and d_type == "object":
        return "entry_path"
    return None


def to_hetero_data(
    scene_graph: dict[str, Any],
) -> tuple[HeteroData, dict[str, list[str]]]:
    """Convert a scene_graph dict into a `(HeteroData, id_order)` pair."""

    nodes = scene_graph.get("nodes", []) or []
    edges = scene_graph.get("edges", []) or []

    by_type: dict[str, list[dict[str, Any]]] = {k: [] for k in KIND_VOCAB}
    id_to_type: dict[str, str] = {}
    for node in nodes:
        ntype = str(node.get("type", "object"))
        if ntype not in by_type:
            ntype = "object"
        by_type[ntype].append(node)
        id_to_type[str(node["id"])] = ntype

    data = HeteroData()
    id_order: dict[str, list[str]] = {}
    for ntype, group in by_type.items():
        if not group:
            continue
        ids = [str(n["id"]) for n in group]
        id_order[ntype] = ids
        x = torch.tensor(
            [_node_feature_vector(n) for n in group], dtype=torch.float32
        )
        label_id = torch.tensor(
            [label_to_index(str(n.get("label", ""))) for n in group],
            dtype=torch.long,
        )
        data[ntype].x = x
        data[ntype].label_id = label_id

    type_pos: dict[str, dict[str, int]] = {
        ntype: {nid: i for i, nid in enumerate(ids)}
        for ntype, ids in id_order.items()
    }

    edge_buckets: dict[
        tuple[str, str, str], list[tuple[int, int, list[float]]]
    ] = {}
    for edge in edges:
        etype = str(edge.get("type", "edge"))
        if _is_excluded_edge_type(etype):
            continue
        src_id = str(edge.get("source", ""))
        dst_id = str(edge.get("target", ""))
        s_type = id_to_type.get(src_id)
        d_type = id_to_type.get(dst_id)
        if s_type is None or d_type is None:
            continue
        bucket = _relation_bucket(s_type, d_type, etype)
        if bucket is None:
            continue
        s_pos = type_pos.get(s_type, {}).get(src_id)
        d_pos = type_pos.get(d_type, {}).get(dst_id)
        if s_pos is None or d_pos is None:
            continue
        edge_buckets.setdefault((s_type, bucket, d_type), []).append(
            (s_pos, d_pos, _edge_attr_row(etype, edge))
        )

    for (s_type, bucket, d_type), triples in edge_buckets.items():
        srcs = torch.tensor([t[0] for t in triples], dtype=torch.long)
        dsts = torch.tensor([t[1] for t in triples], dtype=torch.long)
        feats = torch.tensor([t[2] for t in triples], dtype=torch.float32)
        key = (s_type, bucket, d_type)
        data[key].edge_index = torch.stack([srcs, dsts], dim=0)
        data[key].edge_attr = feats

    data = ToUndirected(merge=False)(data)

    return data, id_order


def hetero_data_summary(data: HeteroData) -> dict[str, Any]:
    out: dict[str, Any] = {"nodes_per_type": {}, "edges_per_relation": {}}
    for ntype in data.node_types:
        out["nodes_per_type"][ntype] = int(data[ntype].x.size(0))
    for rel in data.edge_types:
        out["edges_per_relation"]["__".join(rel)] = int(
            data[rel].edge_index.size(1)
        )
    return out


__all__ = [
    "NODE_FEAT_DIM",
    "CONT_EDGE_DIM",
    "EDGE_FEAT_DIM",
    "EDGE_FEAT_KEYS",
    "EXCLUDED_EDGE_PREFIXES",
    "load_scene_graph",
    "to_hetero_data",
    "hetero_data_summary",
]
