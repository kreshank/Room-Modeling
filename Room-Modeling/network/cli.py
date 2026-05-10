"""Forward-pass CLI for the feng shui GNN.

Reads a `scene_graph.json` produced by the `graph/` pipeline, runs it through
the heterogeneous GAT (random init unless `--weights` is supplied), and emits
predictions in a JSON shape that mirrors the rule engine's `principle_checks`.

Example:

    python -m network.cli predict \\
        --scene_graph Room-Modeling/outs/spatial_editor_outputs/smoke_test/scene_graph.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from .data import hetero_data_summary, load_scene_graph, to_hetero_data
from .labels import (
    HEAD_NODE_TYPES,
    INDEX_TO_STATUS,
    NODE_TYPE_PRINCIPLE_MASK,
    PRINCIPLES,
)
from .model import HeteroGAT, HeteroGATConfig


SCORE_AXIS = torch.tensor([0.0, 0.5, 1.0])  # violated, weak, good


def _build_predictions(
    out: dict[str, torch.Tensor],
    id_order: dict[str, list[str]],
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    score_axis = SCORE_AXIS
    for ntype in HEAD_NODE_TYPES:
        logits = out["node_principle_logits"].get(ntype)
        if logits is None:
            continue
        ids = id_order.get(ntype, [])
        applicable = NODE_TYPE_PRINCIPLE_MASK.get(
            ntype, (False,) * len(PRINCIPLES)
        )
        for n_idx, node_id in enumerate(ids):
            for p_idx, principle in enumerate(PRINCIPLES):
                if not applicable[p_idx]:
                    continue
                p_logits = logits[n_idx, p_idx]
                probs = torch.softmax(p_logits, dim=-1)
                status_idx = int(probs.argmax().item())
                est_score = float((probs * score_axis).sum().item())
                predictions.append(
                    {
                        "principle": principle,
                        "target": node_id,
                        "status": INDEX_TO_STATUS[status_idx],
                        "score": round(est_score, 4),
                    }
                )
    return predictions


def predict(
    scene_graph_path: str | Path,
    weights_path: str | Path | None = None,
    seed: int = 0,
    cfg: HeteroGATConfig | None = None,
) -> dict[str, Any]:
    torch.manual_seed(seed)

    scene_graph = load_scene_graph(scene_graph_path)
    data, id_order = to_hetero_data(scene_graph)

    model = HeteroGAT(cfg)
    if weights_path is not None:
        state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        out = model(data)

    graph_score = float(torch.sigmoid(out["graph_score"]).reshape(-1)[0].item())
    predictions = _build_predictions(out, id_order)
    n_params = sum(p.numel() for p in model.parameters())

    return {
        "schema_version": "fengshui_gnn_v1",
        "source_scene_graph": str(scene_graph_path),
        "weights": str(weights_path) if weights_path else None,
        "graph_summary": hetero_data_summary(data),
        "model_params": n_params,
        "graph_score": round(graph_score, 4),
        "principle_predictions": predictions,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="network",
        description="Forward-pass the feng shui GNN on a scene_graph.json.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_predict = sub.add_parser(
        "predict", help="Run the GNN forward pass on a scene graph."
    )
    p_predict.add_argument(
        "--scene_graph",
        required=True,
        help="Path to scene_graph.json produced by the graph/ pipeline.",
    )
    p_predict.add_argument(
        "--weights",
        default=None,
        help="Optional .pt weights file. Random init if omitted.",
    )
    p_predict.add_argument(
        "--out",
        default=None,
        help="Optional path to write the prediction JSON; otherwise stdout.",
    )
    p_predict.add_argument("--seed", type=int, default=0)

    args = parser.parse_args(argv)

    if args.cmd != "predict":
        parser.error(f"unknown command: {args.cmd}")
        return 2

    scene_path = Path(args.scene_graph).expanduser()
    if not scene_path.exists():
        print(f"Error: scene_graph.json not found: {scene_path}")
        return 2
    weights = Path(args.weights).expanduser() if args.weights else None
    if weights is not None and not weights.exists():
        print(f"Error: weights file not found: {weights}")
        return 2

    result = predict(
        scene_graph_path=scene_path,
        weights_path=weights,
        seed=args.seed,
    )

    text = json.dumps(result, indent=2)
    if args.out:
        out_path = Path(args.out).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"Wrote: {out_path}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
