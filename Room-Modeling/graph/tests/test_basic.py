"""Smoke tests over the bundled SpatialLM sample scene."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph.cli import build_from_scene_json
from graph.config import GraphConfig
from graph.fengshui import evaluate_principles
from graph.functional import apply_functional_layer
from graph.geometry import build_room_geometry
from graph.io import load_scene_json
from graph.scene_graph import (
    DENSE_FEATURE_NAMES,
    ROOM_NODE_ID,
    build_dense_relation_matrix,
    build_scene_graph,
)


SAMPLE_SCENE = (
    Path(__file__).resolve().parents[2]
    / "outs"
    / "spatial_editor_outputs"
    / "my_room"
    / "scene.json"
)


@pytest.fixture(scope="module")
def scene_dict():
    assert SAMPLE_SCENE.exists(), f"Sample scene not found at {SAMPLE_SCENE}"
    return load_scene_json(SAMPLE_SCENE)


@pytest.fixture(scope="module")
def room_geom(scene_dict):
    return build_room_geometry(scene_dict, GraphConfig())


@pytest.fixture(scope="module")
def graph_with_zones(room_geom):
    cfg = GraphConfig()
    graph = build_scene_graph(room_geom, cfg)
    zones, focal = apply_functional_layer(graph, room_geom, cfg)
    return graph, zones, focal


def test_room_polygon_area_in_expected_range(room_geom):
    area = room_geom.room_polygon.area
    assert 15.0 <= area <= 30.0, f"unexpected room area: {area:.2f} m^2"


def test_at_least_one_entry_door(room_geom):
    assert len(room_geom.entry_door_ids) >= 1


def test_walkable_polygon_is_smaller_than_room(room_geom):
    assert room_geom.walkable_polygon.area > 0
    assert room_geom.walkable_polygon.area < room_geom.room_polygon.area


def test_bed_has_backing_to_west_wall(graph_with_zones, room_geom):
    graph, _zones, _focal = graph_with_zones
    bed_id = "bbox_2"
    backings = [
        dst
        for _, dst, data in graph.out_edges(bed_id, data=True)
        if data.get("type") == "has_backing"
    ]
    assert backings, f"bed {bed_id} should have has_backing edges; got {backings}"
    assert "wall_4" in backings, f"expected bed backed by wall_4, got {backings}"


def test_inside_room_edges_attach_to_room_node(graph_with_zones):
    graph, _zones, _focal = graph_with_zones
    inside = [
        (s, d)
        for s, d, data in graph.edges(data=True)
        if data.get("type") == "inside_room"
    ]
    assert inside, "expected inside_room edges"
    for src, dst in inside:
        assert dst == ROOM_NODE_ID


def test_dense_matrix_aligned_with_id_order(room_geom):
    id_order, mat = build_dense_relation_matrix(room_geom)
    assert mat.shape == (len(id_order), len(id_order), len(DENSE_FEATURE_NAMES))
    diag = np.array([mat[i, i, :] for i in range(len(id_order))])
    assert np.allclose(diag, 0.0), "diagonal of dense matrix must be all zeros"

    # Distance feature must be symmetric.
    dist_idx = DENSE_FEATURE_NAMES.index("distance_m")
    for i in range(len(id_order)):
        for j in range(len(id_order)):
            if i == j:
                continue
            assert mat[i, j, dist_idx] == pytest.approx(
                mat[j, i, dist_idx], rel=1e-6, abs=1e-6
            )


def test_command_position_check_emitted_for_bed(graph_with_zones, room_geom):
    graph, zones, _focal = graph_with_zones
    checks = evaluate_principles(graph, room_geom, zones, GraphConfig())
    bed_checks = [
        c for c in checks if c.principle == "command_position" and c.target == "bbox_2"
    ]
    assert bed_checks, "expected a command_position check for the bed"


def test_full_pipeline_via_cli_helper():
    out = build_from_scene_json(SAMPLE_SCENE)
    assert out["schema_version"] == "scene_graph_v1"
    assert out["room"]["area_m2"] > 0
    assert out["nodes"]
    assert out["edges"]
    assert out["dense_relation_matrix"]["shape"][2] == len(DENSE_FEATURE_NAMES)
    assert out["summary"]["n_objects"] >= 1
