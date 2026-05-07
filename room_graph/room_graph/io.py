"""Load `scene.json` produced by `spatiallm_room_editor` and write `scene_graph.json`."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from .geometry import RoomGeometry, polygon_to_xy
from .scene_graph import DENSE_FEATURE_NAMES, ROOM_NODE_ID
from .types import PrincipleCheck, Zone


SCHEMA_VERSION = "scene_graph_v1"


def load_scene_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def graph_to_dict(
    graph: nx.MultiDiGraph,
    room_geom: RoomGeometry,
    zones: list[Zone],
    id_order: list[str],
    dense_matrix: np.ndarray,
    principle_checks: list[PrincipleCheck],
    focal_id: str | None,
    *,
    source_scene_json: str | None = None,
) -> dict[str, Any]:
    nodes_out: list[dict[str, Any]] = []
    for node_id, data in graph.nodes(data=True):
        nodes_out.append(
            {
                "id": node_id,
                "type": data.get("type", "unknown"),
                "kind": data.get("kind", "unknown"),
                "label": data.get("label", ""),
                "geometry": data.get("geometry", {}),
                "attrs": data.get("attrs", {}),
            }
        )

    edges_out: list[dict[str, Any]] = []
    for src, dst, data in graph.edges(data=True):
        edge = {"source": src, "target": dst, "type": data.get("type", "edge")}
        for k, v in data.items():
            if k == "type":
                continue
            edge[k] = v
        edges_out.append(edge)

    walkable_geoms = polygon_to_xy(room_geom.walkable_polygon)

    summary = {
        "n_nodes": graph.number_of_nodes(),
        "n_edges": graph.number_of_edges(),
        "n_objects": len([e for e in room_geom.entities.values() if e.kind in ("object", "furniture")]),
        "n_walls": len(room_geom.walls),
        "n_doors": sum(1 for e in room_geom.entities.values() if e.kind == "door"),
        "n_windows": sum(1 for e in room_geom.entities.values() if e.kind == "window"),
        "n_zones": len(zones),
        "entry_count": len(room_geom.entry_door_ids),
        "passage_count": len(room_geom.passage_door_ids),
        "focal_id": focal_id,
        "principle_violations": sum(
            1 for c in principle_checks if c.status == "violated"
        ),
        "principle_warnings": sum(1 for c in principle_checks if c.status == "weak"),
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "source_scene_json": source_scene_json,
        "room": {
            "polygon_xy": polygon_to_xy(room_geom.room_polygon),
            "walkable_polygon_xy": walkable_geoms,
            "area_m2": float(room_geom.room_polygon.area),
            "centroid_xy": [
                float(room_geom.room_polygon.centroid.x),
                float(room_geom.room_polygon.centroid.y),
            ],
            "entry_door_ids": list(room_geom.entry_door_ids),
            "passage_door_ids": list(room_geom.passage_door_ids),
        },
        "id_order": id_order,
        "nodes": nodes_out,
        "edges": edges_out,
        "zones": [
            {
                "id": z.id,
                "type": z.type,
                "members": list(z.members),
                "centroid_xy": list(z.centroid_xy) if z.centroid_xy else None,
                "attrs": dict(z.attrs),
            }
            for z in zones
        ],
        "dense_relation_matrix": {
            "feature_names": list(DENSE_FEATURE_NAMES),
            "id_order": list(id_order),
            "shape": list(dense_matrix.shape),
            "data": dense_matrix.tolist(),
        },
        "principle_checks": [c.to_dict() for c in principle_checks],
        "summary": summary,
    }


def write_scene_graph_json(scene_graph: dict[str, Any], path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(scene_graph, indent=2), encoding="utf-8")
    return out_path


def write_edges_csv(scene_graph: dict[str, Any], path: str | Path) -> Path:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = scene_graph.get("edges", [])
    fieldnames = ["source", "target", "type"]
    extras: list[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames and k not in extras:
                extras.append(k)
    fieldnames = fieldnames + extras
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out_path
