"""Lightweight metrics for distillation training.

All accumulators take per-scene `(pred, target, mask)` tensors and aggregate
across an epoch. Per-principle macro-F1 averages F1 across the three statuses
(violated / weak / good) for each principle, then reports both the per-
principle vector and the mean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .labels import PRINCIPLES, STATUSES


@dataclass
class StatusF1Accumulator:
    """Per-(principle, status) confusion counts."""

    n_principles: int = len(PRINCIPLES)
    n_statuses: int = len(STATUSES)
    tp: torch.Tensor = field(init=False)
    fp: torch.Tensor = field(init=False)
    fn: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.tp = torch.zeros((self.n_principles, self.n_statuses), dtype=torch.long)
        self.fp = torch.zeros((self.n_principles, self.n_statuses), dtype=torch.long)
        self.fn = torch.zeros((self.n_principles, self.n_statuses), dtype=torch.long)

    def update(
        self,
        pred_status: torch.Tensor,
        target_status: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        """Update counts.

        Shapes: ``(N_nodes, P)`` for each tensor. ``mask`` is a bool tensor.
        """

        if mask.numel() == 0:
            return
        sel = mask.bool()
        if not sel.any():
            return
        pred = pred_status[sel]
        targ = target_status[sel]
        # principle index for each supervised cell, derived from mask column
        _, p_idx = sel.nonzero(as_tuple=True)
        for s in range(self.n_statuses):
            pred_s = pred == s
            targ_s = targ == s
            tp = (pred_s & targ_s).long()
            fp = (pred_s & ~targ_s).long()
            fn = (~pred_s & targ_s).long()
            self.tp[:, s].index_add_(0, p_idx, tp)
            self.fp[:, s].index_add_(0, p_idx, fp)
            self.fn[:, s].index_add_(0, p_idx, fn)

    def per_principle_f1(self) -> torch.Tensor:
        """Macro-averaged F1 (over statuses) per principle. Shape ``(P,)``."""

        precision = self.tp / (self.tp + self.fp).clamp_min(1)
        recall = self.tp / (self.tp + self.fn).clamp_min(1)
        f1_per_status = 2 * precision * recall / (precision + recall).clamp_min(1e-9)
        # If a status never appears (no support) for a principle, exclude it
        # from the macro mean for that principle.
        support = (self.tp + self.fn).clamp_min(0)
        present = (support > 0).float()
        denom = present.sum(dim=1).clamp_min(1)
        return (f1_per_status * present).sum(dim=1) / denom

    def macro_f1(self) -> float:
        per_p = self.per_principle_f1()
        return float(per_p.mean().item())

    def total_supervised(self) -> int:
        return int((self.tp + self.fn).sum().item())


@dataclass
class MaskedMAE:
    sum_abs: float = 0.0
    count: int = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> None:
        sel = mask.bool()
        if not sel.any():
            return
        diff = (pred[sel] - target[sel]).abs()
        self.sum_abs += float(diff.sum().item())
        self.count += int(sel.sum().item())

    def value(self) -> float:
        if self.count == 0:
            return float("nan")
        return self.sum_abs / self.count


def metrics_to_dict(
    status_acc: StatusF1Accumulator,
    score_mae: MaskedMAE,
    graph_mae: MaskedMAE,
) -> dict[str, Any]:
    per_p = status_acc.per_principle_f1().tolist()
    return {
        "macro_f1": status_acc.macro_f1(),
        "per_principle_f1": {
            PRINCIPLES[i]: round(per_p[i], 4) for i in range(len(PRINCIPLES))
        },
        "score_mae": round(score_mae.value(), 4) if score_mae.count else None,
        "graph_score_mae": round(graph_mae.value(), 4) if graph_mae.count else None,
        "supervised_cells": status_acc.total_supervised(),
        "score_cells": score_mae.count,
        "graph_cells": graph_mae.count,
    }


__all__ = [
    "StatusF1Accumulator",
    "MaskedMAE",
    "metrics_to_dict",
]
