"""Distillation training loop for the feng shui GNN.

Each training example is a ``scene_graph.json`` produced by ``graph.cli``.
The model input is the scene graph stripped of ``principle_*`` edges; the
supervision is the rule engine's ``principle_checks`` aligned via
:mod:`network.targets`.

Loss = weighted ``cross_entropy`` on per-node status (ignoring unsupervised
cells) + ``smooth_l1`` on the per-node continuous score over the same cells +
``smooth_l1`` on the per-graph score head.

Batch size is 1 in this milestone — heterogeneous-graph batching with
per-type teacher tensors is straightforward to add later but adds bookkeeping
that is not warranted while the corpus is small.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm.auto import tqdm

from .dataset import FengShuiSceneGraphDataset, resolve_paths
from .labels import HEAD_NODE_TYPES, PRINCIPLES, STATUSES, STATUS_SCORE_AXIS
from .metrics import MaskedMAE, StatusF1Accumulator, metrics_to_dict
from .model import HeteroGAT, HeteroGATConfig
from .targets import IGNORE_INDEX

DEFAULT_RUNS_DIR = Path("outs/network_runs")

SCORE_AXIS = torch.tensor(STATUS_SCORE_AXIS, dtype=torch.float32)


def default_device() -> str:
    """Pick ``cuda`` if available, else ``cpu``. Used as the CLI default."""

    return "cuda" if torch.cuda.is_available() else "cpu"


def _move_targets_to(targets: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move per-type teacher tensors + the graph-level scalar onto ``device``."""

    per_type = targets.get("per_type", {})
    for ntype, tensors in per_type.items():
        for k in ("status", "score", "mask"):
            t = tensors.get(k)
            if isinstance(t, torch.Tensor) and t.device != device:
                tensors[k] = t.to(device)
    g = targets.get("graph_score_target")
    if isinstance(g, torch.Tensor) and g.device != device:
        targets["graph_score_target"] = g.to(device)
    return targets


@dataclass
class TrainConfig:
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    lambda_score: float = 0.5
    lambda_graph: float = 0.25
    seed: int = 0
    device: str = "cpu"


def load_principle_weights(summary_path: str | Path | None) -> torch.Tensor:
    """Build per-principle weights ``w_p ∝ log(1 + count_p)`` from ``tune/``.

    Falls back to uniform weights when the file is missing or no principle
    overlaps the canonical vocabulary.
    """

    weights = torch.ones(len(PRINCIPLES), dtype=torch.float32)
    if summary_path is None:
        return weights
    p = Path(summary_path)
    if not p.exists():
        return weights
    summary = json.loads(p.read_text(encoding="utf-8"))
    counts = summary.get("principle_counts", {}) or {}
    log_counts = []
    for name in PRINCIPLES:
        c = float(counts.get(name, 0))
        log_counts.append(math.log1p(c))
    raw = torch.tensor(log_counts, dtype=torch.float32)
    if float(raw.sum().item()) <= 0.0:
        return weights
    norm = raw / raw.sum() * len(PRINCIPLES)
    norm = norm.clamp_min(0.1)
    return norm


def _expected_score(logits: torch.Tensor, score_axis: torch.Tensor) -> torch.Tensor:
    """Probability-weighted scalar score per ``(node, principle)`` cell."""

    probs = torch.softmax(logits, dim=-1)
    return (probs * score_axis).sum(dim=-1)


def compute_losses(
    model_out: dict[str, Any],
    targets: dict[str, Any],
    *,
    principle_weights: torch.Tensor,
    cfg: TrainConfig,
    score_axis: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return scalar loss + a dict of detached component values."""

    parts: dict[str, float] = {}
    total = torch.zeros((), dtype=torch.float32, device=principle_weights.device)
    n_components = 0

    for ntype in HEAD_NODE_TYPES:
        logits = model_out["node_principle_logits"].get(ntype)
        if logits is None:
            continue
        teacher = targets["per_type"].get(ntype)
        if teacher is None:
            continue
        status_t = teacher["status"]
        score_t = teacher["score"]
        mask = teacher["mask"]
        if not mask.any():
            continue

        n_nodes, n_p, n_s = logits.shape

        # Status CE: flatten to (N*P, n_statuses) with ignore_index masking.
        flat_logits = logits.reshape(-1, n_s)
        flat_status = status_t.reshape(-1)
        per_cell_w = principle_weights.unsqueeze(0).expand(n_nodes, n_p).reshape(-1)
        ce_per = F.cross_entropy(
            flat_logits, flat_status, ignore_index=IGNORE_INDEX, reduction="none"
        )
        weighted = ce_per * per_cell_w
        valid = (flat_status != IGNORE_INDEX).float()
        denom = valid.sum().clamp_min(1.0)
        ce_loss = weighted.sum() / denom
        total = total + ce_loss
        parts[f"ce_{ntype}"] = float(ce_loss.detach().item())
        n_components += 1

        # Score smooth-L1 on supervised cells.
        if cfg.lambda_score > 0:
            est_score = _expected_score(logits, score_axis)
            sel = mask.bool()
            if sel.any():
                score_loss = F.smooth_l1_loss(
                    est_score[sel], score_t[sel].nan_to_num(0.0)
                )
                total = total + cfg.lambda_score * score_loss
                parts[f"score_{ntype}"] = float(score_loss.detach().item())

    # Per-graph head.
    graph_pred_logit = model_out.get("graph_score")
    graph_target = targets.get("graph_score_target")
    if (
        cfg.lambda_graph > 0
        and graph_pred_logit is not None
        and graph_target is not None
        and not torch.isnan(graph_target).item()
    ):
        graph_pred = torch.sigmoid(graph_pred_logit.reshape(()))
        graph_loss = F.smooth_l1_loss(graph_pred, graph_target)
        total = total + cfg.lambda_graph * graph_loss
        parts["graph_score"] = float(graph_loss.detach().item())

    if n_components == 0 and "graph_score" not in parts:
        return total, parts
    return total, parts


def evaluate(
    model: HeteroGAT,
    dataset: FengShuiSceneGraphDataset,
    *,
    device: str,
    desc: str = "eval",
    leave: bool = False,
) -> dict[str, Any]:
    model.eval()
    score_axis = SCORE_AXIS.to(device)
    status_acc = StatusF1Accumulator()
    score_mae = MaskedMAE()
    graph_mae = MaskedMAE()

    dev = torch.device(device)
    with torch.no_grad():
        iterator = tqdm(
            dataset,
            total=len(dataset),
            desc=desc,
            leave=leave,
            dynamic_ncols=True,
            unit="graph",
        )
        for data, targets, _id_order, _path in iterator:
            data = data.to(dev)
            out = model(data)
            for ntype in HEAD_NODE_TYPES:
                logits = out["node_principle_logits"].get(ntype)
                if logits is None:
                    continue
                teacher = targets["per_type"].get(ntype)
                if teacher is None or not teacher["mask"].any():
                    continue
                # Metric accumulators run on CPU; move predictions, leave
                # teacher tensors on CPU (cheap; small per-graph buffers).
                pred_status = logits.argmax(dim=-1).cpu()
                est_score = _expected_score(logits, score_axis).cpu()
                status_acc.update(
                    pred_status, teacher["status"].cpu(), teacher["mask"].cpu()
                )
                score_t = teacher["score"].cpu().nan_to_num(0.0)
                score_mae.update(est_score, score_t, teacher["mask"].cpu())

            graph_pred = torch.sigmoid(out["graph_score"].reshape(())).cpu()
            graph_target = targets.get("graph_score_target")
            if graph_target is not None and not torch.isnan(graph_target).item():
                graph_mae.update(
                    graph_pred.unsqueeze(0),
                    graph_target.reshape(1).cpu(),
                    torch.ones(1, dtype=torch.bool),
                )

    return metrics_to_dict(status_acc, score_mae, graph_mae)


def _format_epoch_line(record: dict[str, Any], *, total_epochs: int, marker: str = "") -> str:
    """Return a compact single-line human summary of one epoch."""

    epoch = record["epoch"]
    width = max(2, len(str(total_epochs)))
    bits = [f"epoch {epoch:>{width}}/{total_epochs}"]
    bits.append(f"loss={record.get('train_loss', float('nan')):.4f}")
    val = record.get("val") or {}
    if val:
        f1 = val.get("macro_f1")
        if isinstance(f1, (int, float)) and not math.isnan(float(f1)):
            bits.append(f"val_f1={float(f1):.4f}")
        smae = val.get("score_mae")
        if isinstance(smae, (int, float)) and not math.isnan(float(smae)):
            bits.append(f"score_mae={float(smae):.4f}")
        gmae = val.get("graph_score_mae")
        if isinstance(gmae, (int, float)) and not math.isnan(float(gmae)):
            bits.append(f"graph_mae={float(gmae):.4f}")
    line = " | ".join(bits)
    return f"{line}{marker}"


def train(
    *,
    train_paths: Sequence[Path],
    val_paths: Sequence[Path] | None,
    cfg: TrainConfig,
    model_cfg: HeteroGATConfig | None = None,
    summary_path: str | Path | None = None,
    checkpoint_dir: str | Path | None = None,
) -> dict[str, Any]:
    if not train_paths:
        raise ValueError("No training paths resolved")
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device)
    train_ds = FengShuiSceneGraphDataset(train_paths)
    val_ds = (
        FengShuiSceneGraphDataset(val_paths) if val_paths else None
    )

    model = HeteroGAT(model_cfg).to(device)
    optim = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    weights = load_principle_weights(summary_path).to(device)
    score_axis = SCORE_AXIS.to(device)

    history: list[dict[str, Any]] = []
    best_metric = -math.inf
    best_path: Path | None = None
    log_path: Path | None = None
    if checkpoint_dir is not None:
        ckpt_dir = Path(checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_path = ckpt_dir / "log.jsonl"
        # Truncate any prior log so each run starts with a clean JSONL stream.
        log_path.write_text("", encoding="utf-8")

    indices = list(range(len(train_ds)))
    epoch_bar = tqdm(
        range(1, cfg.epochs + 1),
        desc="train",
        unit="epoch",
        dynamic_ncols=True,
    )
    for epoch in epoch_bar:
        model.train()
        random.shuffle(indices)
        epoch_loss = 0.0
        n_examples = 0
        component_sums: dict[str, float] = {}

        inner_bar = tqdm(
            indices,
            desc=f"epoch {epoch}/{cfg.epochs}",
            leave=False,
            dynamic_ncols=True,
            unit="graph",
        )
        for i in inner_bar:
            data, targets, _id_order, _path = train_ds[i]
            data = data.to(device)
            targets = _move_targets_to(targets, device)
            optim.zero_grad()
            out = model(data)
            loss, parts = compute_losses(
                out,
                targets,
                principle_weights=weights,
                cfg=cfg,
                score_axis=score_axis,
            )
            if loss.requires_grad:
                loss.backward()
                optim.step()
                step_loss = float(loss.detach().item())
                epoch_loss += step_loss
                n_examples += 1
                for k, v in parts.items():
                    component_sums[k] = component_sums.get(k, 0.0) + v
                inner_bar.set_postfix(
                    loss=f"{epoch_loss / n_examples:.4f}",
                    refresh=False,
                )
        inner_bar.close()

        avg_loss = epoch_loss / max(1, n_examples)
        record: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": round(avg_loss, 6),
            "train_examples": n_examples,
            "components": {
                k: round(v / max(1, n_examples), 6)
                for k, v in component_sums.items()
            },
        }
        marker = ""
        if val_ds is not None and len(val_ds) > 0:
            val_metrics = evaluate(
                model,
                val_ds,
                device=cfg.device,
                desc=f"val {epoch}/{cfg.epochs}",
                leave=False,
            )
            record["val"] = val_metrics
            current = val_metrics.get("macro_f1", float("nan"))
            if isinstance(current, float) and not math.isnan(current):
                if current > best_metric and checkpoint_dir is not None:
                    best_metric = current
                    best_path = Path(checkpoint_dir) / "best.pt"
                    torch.save(model.state_dict(), best_path)
                    record["checkpoint"] = str(best_path)
                    marker = "  *best (saved)"
        history.append(record)

        line = _format_epoch_line(record, total_epochs=cfg.epochs, marker=marker)
        tqdm.write(line)
        if log_path is not None:
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")

        epoch_bar.set_postfix(
            loss=f"{avg_loss:.4f}",
            best_f1=(
                f"{best_metric:.4f}" if best_metric > -math.inf else "n/a"
            ),
            refresh=False,
        )
    epoch_bar.close()

    if checkpoint_dir is not None:
        last_path = Path(checkpoint_dir) / "last.pt"
        torch.save(model.state_dict(), last_path)

    return {
        "history": history,
        "best_macro_f1": best_metric if best_metric > -math.inf else None,
        "best_checkpoint": str(best_path) if best_path else None,
    }


def add_train_arguments(parser: Any) -> None:
    parser.add_argument("--train-glob", action="append", default=[],
                        help="Glob for training scene_graph.json files (repeatable).")
    parser.add_argument("--val-glob", action="append", default=[],
                        help="Glob for validation scene_graph.json files (repeatable).")
    parser.add_argument("--train-manifest", default=None,
                        help="Optional newline-separated paths or jsonl with `scene_graph` keys.")
    parser.add_argument("--val-manifest", default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-score", type=float, default=0.5)
    parser.add_argument("--lambda-graph", type=float, default=0.25)
    parser.add_argument("--summary-json", default=None,
                        help="Path to outs/transcripts/summary.json for principle weights.")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Output directory for best.pt/last.pt/log.jsonl/history.json. "
             f"Defaults to {DEFAULT_RUNS_DIR}/train_<timestamp>/ so iterative "
             "runs do not clobber each other.",
    )
    parser.add_argument(
        "--device",
        default=default_device(),
        help="Defaults to 'cuda' if available, else 'cpu'. "
             "Force CPU with --device cpu, or pin a GPU with --device cuda:1.",
    )
    parser.add_argument("--seed", type=int, default=0)


def _resolve_checkpoint_dir(provided: str | None) -> Path:
    if provided:
        return Path(provided).expanduser()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return DEFAULT_RUNS_DIR / f"train_{timestamp}"


def run_train(args: Any) -> int:
    train_paths = resolve_paths(globs=args.train_glob, manifest=args.train_manifest)
    val_paths = resolve_paths(globs=args.val_glob, manifest=args.val_manifest)
    if not train_paths:
        print("Error: no training graphs resolved.")
        return 2

    cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lambda_score=args.lambda_score,
        lambda_graph=args.lambda_graph,
        seed=args.seed,
        device=args.device,
    )
    checkpoint_dir = _resolve_checkpoint_dir(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print("Training fengshui GNN")
    print(f"  device:       {cfg.device}")
    print(f"  train graphs: {len(train_paths)}")
    print(f"  val   graphs: {len(val_paths)}")
    print(f"  epochs:       {cfg.epochs}  (lr={cfg.lr}, wd={cfg.weight_decay})")
    if args.summary_json:
        print(f"  weights from: {args.summary_json}")
    print(f"  output dir:   {checkpoint_dir}")
    print(f"  log:          {checkpoint_dir / 'log.jsonl'}")
    print()

    result = train(
        train_paths=train_paths,
        val_paths=val_paths,
        cfg=cfg,
        summary_path=args.summary_json,
        checkpoint_dir=checkpoint_dir,
    )

    history_path = checkpoint_dir / "history.json"
    history_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print()
    best = result.get("best_macro_f1")
    best_ckpt = result.get("best_checkpoint")
    if best is not None and best_ckpt:
        print(f"Best val macro_f1: {float(best):.4f}  ->  {best_ckpt}")
    elif val_paths:
        print("No improvement recorded on validation set.")
    print(f"Last weights:      {checkpoint_dir / 'last.pt'}")
    print(f"Per-epoch log:     {checkpoint_dir / 'log.jsonl'}")
    print(f"Training summary:  {history_path}")
    return 0


__all__ = [
    "TrainConfig",
    "DEFAULT_RUNS_DIR",
    "default_device",
    "load_principle_weights",
    "compute_losses",
    "evaluate",
    "train",
    "add_train_arguments",
    "run_train",
]
