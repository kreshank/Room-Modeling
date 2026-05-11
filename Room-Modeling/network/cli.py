"""CLI for the feng shui GNN: ``predict``, ``train``, ``eval``."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from .data import hetero_data_summary, load_scene_graph, to_hetero_data
from .eval import add_eval_arguments, run_eval
from .labels import (
    HEAD_NODE_TYPES,
    INDEX_TO_STATUS,
    NODE_TYPE_PRINCIPLE_MASK,
    PRINCIPLES,
    STATUS_SCORE_AXIS,
)
from .model import HeteroGAT, HeteroGATConfig
from .train import add_train_arguments, run_train

DEFAULT_INFERENCE_DIR = Path("outs/inference")
SCORE_AXIS = torch.tensor(STATUS_SCORE_AXIS, dtype=torch.float32)


def _build_predictions(
    out: dict[str, torch.Tensor],
    id_order: dict[str, list[str]],
) -> list[dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
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
                est_score = float((probs * SCORE_AXIS).sum().item())
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
        description="Feng shui GNN: predict, train, or eval on scene graphs.",
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
        help="Path to write the prediction JSON. Defaults to "
             f"{DEFAULT_INFERENCE_DIR}/predict_<scene>_<timestamp>.json.",
    )
    p_predict.add_argument("--seed", type=int, default=0)

    p_train = sub.add_parser(
        "train", help="Distill the rule engine into the GNN."
    )
    add_train_arguments(p_train)

    p_eval = sub.add_parser(
        "eval", help="Score a checkpoint against the rule engine."
    )
    add_eval_arguments(p_eval)

    args = parser.parse_args(argv)

    if args.cmd == "train":
        return run_train(args)
    if args.cmd == "eval":
        return run_eval(args)

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

    out_path = _resolve_predict_out_path(args.out, scene_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    _print_predict_summary(result)
    print(f"Predictions JSON: {out_path}")
    return 0


def _resolve_predict_out_path(provided: str | None, scene_path: Path) -> Path:
    if provided:
        return Path(provided).expanduser()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", scene_path.parent.name or "scene")
    return DEFAULT_INFERENCE_DIR / f"predict_{safe_stem}_{timestamp}.json"


def _print_predict_summary(result: dict[str, Any]) -> None:
    preds = result.get("principle_predictions", []) or []
    status_counts = Counter(p.get("status", "?") for p in preds)
    type_counts = Counter()
    for ntype, n in (result.get("graph_summary", {}).get("nodes_per_type") or {}).items():
        type_counts[ntype] = int(n)
    print("Predict")
    print(f"  source:      {result.get('source_scene_graph')}")
    print(f"  weights:     {result.get('weights') or '(random init)'}")
    print(f"  model_params:{result.get('model_params'):>10,}")
    score = result.get("graph_score")
    if isinstance(score, (int, float)):
        print(f"  graph_score: {float(score):.4f}")
    if type_counts:
        nodes_part = ", ".join(f"{k}={v}" for k, v in sorted(type_counts.items()))
        print(f"  nodes:       {nodes_part}")
    if preds:
        statuses = ", ".join(f"{k}={status_counts[k]}" for k in ("good", "weak", "violated") if k in status_counts)
        print(f"  predictions: {len(preds)} cells ({statuses})")
    else:
        print("  predictions: 0 cells")


if __name__ == "__main__":
    raise SystemExit(main())
