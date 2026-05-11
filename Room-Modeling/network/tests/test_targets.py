"""Targets / dataset alignment against the smoke_test fixture."""

from __future__ import annotations

import math
from pathlib import Path

import torch

from network.data import load_scene_graph, to_hetero_data
from network.labels import HEAD_NODE_TYPES, PRINCIPLES, STATUS_TO_INDEX
from network.targets import IGNORE_INDEX, build_teacher_tensors


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_PATH = REPO_ROOT / "outs" / "spatial_editor_outputs" / "smoke_test" / "scene_graph.json"


def test_smoke_fixture_present():
    assert SMOKE_PATH.exists(), f"missing fixture: {SMOKE_PATH}"


def test_supervised_cells_match_principle_checks():
    scene_graph = load_scene_graph(SMOKE_PATH)
    _data, id_order = to_hetero_data(scene_graph)
    targets = build_teacher_tensors(scene_graph, id_order)

    expected = 0
    for ck in scene_graph.get("principle_checks", []):
        if ck.get("principle") not in PRINCIPLES:
            continue
        if ck.get("status") not in STATUS_TO_INDEX:
            continue
        target = ck.get("target")
        owns = any(target in id_order.get(nt, []) for nt in HEAD_NODE_TYPES)
        if owns:
            expected += 1

    actual = sum(
        int(targets["per_type"][nt]["mask"].sum().item())
        for nt in HEAD_NODE_TYPES
        if nt in targets["per_type"]
    )
    assert actual == expected
    assert actual > 0


def test_unsupervised_cells_use_ignore_index():
    scene_graph = load_scene_graph(SMOKE_PATH)
    _data, id_order = to_hetero_data(scene_graph)
    targets = build_teacher_tensors(scene_graph, id_order)
    for nt in HEAD_NODE_TYPES:
        per = targets["per_type"].get(nt)
        if per is None:
            continue
        unsup = ~per["mask"]
        if unsup.any():
            assert (per["status"][unsup] == IGNORE_INDEX).all()
            assert torch.isnan(per["score"][unsup]).all()


def test_specific_known_check_lines_up():
    """The smoke fixture has `bbox_2 / command_position / violated / 0.4`."""

    scene_graph = load_scene_graph(SMOKE_PATH)
    _data, id_order = to_hetero_data(scene_graph)
    targets = build_teacher_tensors(scene_graph, id_order)

    obj_ids = id_order["object"]
    n_idx = obj_ids.index("bbox_2")
    p_idx = PRINCIPLES.index("command_position")
    obj = targets["per_type"]["object"]

    assert obj["mask"][n_idx, p_idx].item() is True
    assert int(obj["status"][n_idx, p_idx]) == STATUS_TO_INDEX["violated"]
    assert math.isclose(float(obj["score"][n_idx, p_idx]), 0.4, abs_tol=1e-6)


def test_graph_score_is_mean_of_check_scores():
    scene_graph = load_scene_graph(SMOKE_PATH)
    _data, id_order = to_hetero_data(scene_graph)
    targets = build_teacher_tensors(scene_graph, id_order)
    expected = sum(float(c["score"]) for c in scene_graph["principle_checks"]) / len(
        scene_graph["principle_checks"]
    )
    assert math.isclose(float(targets["graph_score_target"]), expected, abs_tol=1e-6)
