"""Command-line entry point: scene.json -> scene_graph.json + edges.csv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from .config import GraphConfig
from .fengshui import attach_principle_edges, evaluate_principles
from .functional import apply_functional_layer
from .geometry import build_room_geometry
from .io import (
    graph_to_dict,
    load_scene_json,
    write_edges_csv,
    write_scene_graph_json,
)
from .scene_graph import build_dense_relation_matrix, build_scene_graph


def build_from_scene_json(
    scene_path: str | Path, cfg: GraphConfig | None = None
) -> dict:
    scene_path = Path(scene_path)
    scene = load_scene_json(scene_path)
    cfg = cfg or GraphConfig()

    room_geom = build_room_geometry(scene, cfg)
    graph = build_scene_graph(room_geom, cfg)
    zones, focal_id = apply_functional_layer(graph, room_geom, cfg)
    id_order, dense_matrix = build_dense_relation_matrix(room_geom)

    checks = evaluate_principles(graph, room_geom, zones, cfg)
    attach_principle_edges(graph, checks)

    return graph_to_dict(
        graph,
        room_geom,
        zones,
        id_order,
        dense_matrix,
        checks,
        focal_id,
        source_scene_json=str(scene_path),
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a deterministic 3D scene graph from a SpatialLM scene.json"
    )
    parser.add_argument("--scene", required=True, help="Path to scene.json from spatiallm_room_editor")
    parser.add_argument(
        "--out",
        default=None,
        help="Output scene_graph.json path (default: alongside scene.json)",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional output edges.csv path (default: alongside scene_graph.json)",
    )
    parser.add_argument(
        "--no_csv",
        action="store_true",
        help="Skip writing the edges.csv companion file",
    )
    args = parser.parse_args(argv)

    scene_path = Path(args.scene).expanduser().resolve()
    if not scene_path.exists():
        print(f"Error: scene.json not found: {scene_path}", file=sys.stderr)
        return 2

    out_path = (
        Path(args.out).expanduser().resolve()
        if args.out
        else scene_path.with_name("scene_graph.json")
    )

    scene_graph = build_from_scene_json(scene_path)
    write_scene_graph_json(scene_graph, out_path)

    if not args.no_csv:
        csv_path = (
            Path(args.csv).expanduser().resolve()
            if args.csv
            else out_path.with_name("edges.csv")
        )
        write_edges_csv(scene_graph, csv_path)
        print(f"Wrote: {csv_path}")

    print(f"Wrote: {out_path}")
    summary = scene_graph.get("summary", {})
    print(
        "Summary: "
        f"{summary.get('n_nodes', 0)} nodes, {summary.get('n_edges', 0)} edges, "
        f"{summary.get('n_zones', 0)} zones, "
        f"{summary.get('principle_violations', 0)} violations, "
        f"{summary.get('principle_warnings', 0)} warnings"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
