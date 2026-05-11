"""Teacher-label tensors derived from a scene graph's `principle_checks`.

The rule engine in `graph/fengshui.py` emits a sparse list of
`PrincipleCheck` records per scene. This module aligns those records with the
per-type node ordering produced by `network.data.to_hetero_data` so the model's
heads can be supervised cell-by-cell.

For each head node type ``ntype`` in :data:`HEAD_NODE_TYPES`, we build:

* ``status``: ``LongTensor (N_ntype, P)``; cells with no teacher row are filled
  with :data:`IGNORE_INDEX` (= ``-1``) so a vanilla ``cross_entropy`` call with
  ``ignore_index=-1`` skips them.
* ``score``: ``FloatTensor (N_ntype, P)``; same shape, ``NaN`` where ignored.
* ``mask``: ``BoolTensor (N_ntype, P)``; ``True`` where a teacher row exists.

We additionally return a single graph-level scalar ``graph_score_target`` —
the unweighted mean of ``check.score`` over the entire scene, which is what
the per-graph regression head is asked to imitate.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from .labels import (
    HEAD_NODE_TYPES,
    PRINCIPLES,
    STATUS_TO_INDEX,
)


IGNORE_INDEX: int = -1


def _principle_index(name: str) -> int | None:
    try:
        return PRINCIPLES.index(name)
    except ValueError:
        return None


def build_teacher_tensors(
    scene_graph: dict[str, Any],
    id_order: dict[str, list[str]],
) -> dict[str, Any]:
    """Return per-type teacher tensors plus a graph-level scalar.

    Parameters
    ----------
    scene_graph
        The dict loaded from a ``scene_graph.json`` file.
    id_order
        The ``id_order`` mapping returned alongside ``HeteroData`` from
        :func:`network.data.to_hetero_data`. Used so target indices line up
        with model output indices for each node type.
    """

    n_principles = len(PRINCIPLES)
    type_pos: dict[str, dict[str, int]] = {
        ntype: {nid: i for i, nid in enumerate(ids)}
        for ntype, ids in id_order.items()
    }

    per_type: dict[str, dict[str, torch.Tensor]] = {}
    for ntype in HEAD_NODE_TYPES:
        n_nodes = len(id_order.get(ntype, []))
        per_type[ntype] = {
            "status": torch.full(
                (n_nodes, n_principles), IGNORE_INDEX, dtype=torch.long
            ),
            "score": torch.full(
                (n_nodes, n_principles), float("nan"), dtype=torch.float32
            ),
            "mask": torch.zeros((n_nodes, n_principles), dtype=torch.bool),
        }

    score_sum = 0.0
    score_count = 0

    for check in scene_graph.get("principle_checks", []) or []:
        principle = str(check.get("principle", ""))
        target = str(check.get("target", ""))
        status = str(check.get("status", ""))
        score_raw = check.get("score", None)

        p_idx = _principle_index(principle)
        if p_idx is None:
            continue
        s_idx = STATUS_TO_INDEX.get(status)
        if s_idx is None:
            continue

        owning_type: str | None = None
        for ntype in HEAD_NODE_TYPES:
            if target in type_pos.get(ntype, {}):
                owning_type = ntype
                break
        if owning_type is None:
            continue
        n_idx = type_pos[owning_type][target]

        per_type[owning_type]["status"][n_idx, p_idx] = s_idx
        per_type[owning_type]["mask"][n_idx, p_idx] = True

        try:
            score_val = float(score_raw)
        except (TypeError, ValueError):
            score_val = float("nan")
        per_type[owning_type]["score"][n_idx, p_idx] = score_val
        if not math.isnan(score_val):
            score_sum += score_val
            score_count += 1

    if score_count > 0:
        graph_score = torch.tensor(score_sum / score_count, dtype=torch.float32)
    else:
        graph_score = torch.tensor(float("nan"), dtype=torch.float32)

    return {
        "per_type": per_type,
        "graph_score_target": graph_score,
    }


__all__ = [
    "IGNORE_INDEX",
    "build_teacher_tensors",
]
