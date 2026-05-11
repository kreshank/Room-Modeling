"""Evaluation entry point: report agreement with the rule engine."""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

from .dataset import FengShuiSceneGraphDataset, resolve_paths
from .model import HeteroGAT, HeteroGATConfig
from .train import DEFAULT_RUNS_DIR, default_device, evaluate


def add_eval_arguments(parser: Any) -> None:
    parser.add_argument("--checkpoint", required=False, default=None,
                        help="Path to a .pt state_dict. Random init when omitted.")
    parser.add_argument("--glob", action="append", default=[],
                        help="Glob for evaluation scene_graph.json files (repeatable).")
    parser.add_argument("--manifest", default=None)
    parser.add_argument(
        "--device",
        default=default_device(),
        help="Defaults to 'cuda' if available, else 'cpu'. "
             "Force CPU with --device cpu, or pin a GPU with --device cuda:1.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Path to write metrics.json. Defaults to "
             f"{DEFAULT_RUNS_DIR}/eval_<timestamp>/metrics.json so runs do not "
             "overwrite each other.",
    )


def _resolve_out_path(provided: str | None) -> Path:
    if provided:
        return Path(provided).expanduser()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return DEFAULT_RUNS_DIR / f"eval_{timestamp}" / "metrics.json"


def _format_float(value: Any, *, digits: int = 4) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if math.isnan(x):
        return "n/a"
    return f"{x:.{digits}f}"


def _print_eval_summary(metrics: dict[str, Any], *, n_graphs: int) -> None:
    print("Evaluation summary")
    ckpt = metrics.get("checkpoint") or "(random init)"
    print(f"  checkpoint:      {ckpt}")
    print(f"  graphs scored:   {n_graphs}")
    print(f"  macro_f1:        {_format_float(metrics.get('macro_f1'))}")
    print(f"  score_mae:       {_format_float(metrics.get('score_mae'))}")
    print(f"  graph_score_mae: {_format_float(metrics.get('graph_score_mae'))}")
    print(f"  supervised cells: {metrics.get('supervised_cells', 0)}  "
          f"(score cells: {metrics.get('score_cells', 0)}, "
          f"graph cells: {metrics.get('graph_cells', 0)})")
    per = metrics.get("per_principle_f1") or {}
    if per:
        print("  per-principle F1:")
        width = max((len(name) for name in per), default=0)
        for name, value in per.items():
            print(f"    {name:<{width}}  {_format_float(value)}")


def run_eval(args: Any) -> int:
    paths = resolve_paths(globs=args.glob, manifest=args.manifest)
    if not paths:
        print("Error: no evaluation graphs resolved.")
        return 2

    device = torch.device(args.device)
    model = HeteroGAT(HeteroGATConfig()).to(device)
    if args.checkpoint:
        ckpt = Path(args.checkpoint)
        if not ckpt.exists():
            print(f"Error: checkpoint not found: {ckpt}")
            return 2
        state = torch.load(str(ckpt), map_location=device, weights_only=True)
        model.load_state_dict(state)
    dataset = FengShuiSceneGraphDataset(paths)

    print(f"Evaluating {len(dataset)} graph(s) on {args.device}.")
    metrics = evaluate(model, dataset, device=args.device, desc="eval", leave=False)
    metrics["checkpoint"] = args.checkpoint
    metrics["n_graphs"] = len(dataset)

    out_path = _resolve_out_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print()
    _print_eval_summary(metrics, n_graphs=len(dataset))
    print()
    print(f"Metrics JSON: {out_path}")
    return 0


__all__ = [
    "add_eval_arguments",
    "run_eval",
]
