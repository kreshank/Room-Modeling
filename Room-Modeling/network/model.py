"""Heterogeneous GAT for feng shui principle-contribution prediction.

Architecture overview:

* Per-node-type input MLP that fuses 8-d pose/size features with a learned
  label embedding.
* `num_layers` x `HeteroConv({rel: GATv2Conv(...)})` with a residual + LayerNorm
  per node type around each layer (default **2** layers).
* Per-node multi-label head sized `n_principles * n_statuses` for each node
  type that has at least one applicable principle.
* Per-graph score head over a (mean, max)-pooled concatenation of all node
  embeddings.

Lightweight by design: at default settings the parameter count is comfortably
under the 250k budget specified in the plan.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATv2Conv, HeteroConv

from .data import EDGE_FEAT_DIM, NODE_FEAT_DIM
from .labels import (
    HEAD_NODE_TYPES,
    KIND_VOCAB,
    LABEL_VOCAB_SIZE,
    NODE_TYPE_PRINCIPLE_MASK,
    PRINCIPLES,
    STATUSES,
)


# Forward-only buckets produced by `data.to_hetero_data` **before**
# `ToUndirected`.  After conversion, PyG adds reverse relations named
# ``(dst, rev_<bucket>, src)`` — see ``default_metadata``.
_FORWARD_RELATIONS: tuple[tuple[str, str, str], ...] = (
    ("object", "spatial", "object"),
    ("object", "to_wall", "wall"),
    ("door", "aperture_wall", "wall"),
    ("window", "aperture_wall", "wall"),
    ("object", "to_room", "room"),
    ("door", "to_room", "room"),
    ("window", "to_room", "room"),
    ("object", "to_zone", "zone"),
    ("door", "door_to_zone", "zone"),
    ("door", "entry_path", "object"),
)


def default_metadata() -> tuple[list[str], list[tuple[str, str, str]]]:
    """Node types + **all** heterogeneous edge keys present after ``ToUndirected``.

    PyG names reverse edges ``(dst_type, 'rev_<bucket>', src_type)`` — including
    self-loop buckets like ``(object, rev_spatial, object)``.
    """

    relations: list[tuple[str, str, str]] = []
    for s_type, bucket, d_type in _FORWARD_RELATIONS:
        relations.append((s_type, bucket, d_type))
        if s_type == d_type:
            relations.append((s_type, f"rev_{bucket}", d_type))
        else:
            relations.append((d_type, f"rev_{bucket}", s_type))
    return list(KIND_VOCAB), relations


@dataclass
class HeteroGATConfig:
    hidden: int = 32
    heads: int = 4
    num_layers: int = 2
    label_emb_dim: int = 12
    dropout: float = 0.1


class HeteroGAT(nn.Module):
    """Heterogeneous graph attention network with principle and score heads."""

    def __init__(self, cfg: HeteroGATConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or HeteroGATConfig()
        node_types, relations = default_metadata()
        self.node_types: list[str] = node_types
        self.relations: list[tuple[str, str, str]] = relations

        h = self.cfg.hidden
        emb_dim = self.cfg.label_emb_dim
        if h % self.cfg.heads != 0:
            raise ValueError(
                f"hidden ({h}) must be divisible by heads ({self.cfg.heads})"
            )

        self.label_emb = nn.Embedding(LABEL_VOCAB_SIZE, emb_dim)

        self.input_proj = nn.ModuleDict(
            {
                ntype: nn.Sequential(
                    nn.Linear(NODE_FEAT_DIM + emb_dim, h),
                    nn.GELU(),
                    nn.LayerNorm(h),
                )
                for ntype in node_types
            }
        )

        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        for _ in range(self.cfg.num_layers):
            convs_for_layer = {
                rel: GATv2Conv(
                    in_channels=h,
                    out_channels=h // self.cfg.heads,
                    heads=self.cfg.heads,
                    edge_dim=EDGE_FEAT_DIM,
                    add_self_loops=False,
                    dropout=self.cfg.dropout,
                )
                for rel in relations
            }
            self.convs.append(HeteroConv(convs_for_layer, aggr="sum"))
            self.layer_norms.append(
                nn.ModuleDict({nt: nn.LayerNorm(h) for nt in node_types})
            )

        n_principles = len(PRINCIPLES)
        n_statuses = len(STATUSES)

        self.principle_heads = nn.ModuleDict(
            {
                ntype: nn.Linear(h, n_principles * n_statuses)
                for ntype in HEAD_NODE_TYPES
            }
        )
        self.score_head = nn.Sequential(
            nn.Linear(h * 2, h),
            nn.GELU(),
            nn.Linear(h, 1),
        )

    def encode(self, data: HeteroData) -> dict[str, torch.Tensor]:
        x_dict: dict[str, torch.Tensor] = {}
        for ntype in self.node_types:
            if ntype not in data.node_types:
                continue
            feats = data[ntype].x
            label_emb = self.label_emb(data[ntype].label_id)
            inp = torch.cat([feats, label_emb], dim=-1)
            x_dict[ntype] = self.input_proj[ntype](inp)

        edge_index_dict: dict[tuple[str, str, str], torch.Tensor] = {}
        edge_attr_dict: dict[tuple[str, str, str], torch.Tensor] = {}
        for rel in data.edge_types:
            edge_index_dict[rel] = data[rel].edge_index
            attr = getattr(data[rel], "edge_attr", None)
            if attr is None:
                attr = torch.zeros(
                    (edge_index_dict[rel].size(1), EDGE_FEAT_DIM),
                    dtype=torch.float32,
                    device=edge_index_dict[rel].device,
                )
            edge_attr_dict[rel] = attr

        for layer_idx, conv in enumerate(self.convs):
            out = conv(
                x_dict,
                edge_index_dict,
                edge_attr_dict=edge_attr_dict,
            )
            ln = self.layer_norms[layer_idx]
            new_x_dict: dict[str, torch.Tensor] = {}
            for ntype, h in x_dict.items():
                if ntype in out:
                    new_x_dict[ntype] = ln[ntype](h + F.gelu(out[ntype]))
                else:
                    new_x_dict[ntype] = ln[ntype](h)
            x_dict = new_x_dict

        return x_dict

    def forward(self, data: HeteroData) -> dict[str, torch.Tensor]:
        x_dict = self.encode(data)

        node_principle_logits: dict[str, torch.Tensor] = {}
        for ntype in HEAD_NODE_TYPES:
            if ntype not in x_dict:
                continue
            h = x_dict[ntype]
            logits = self.principle_heads[ntype](h)
            logits = logits.view(h.size(0), len(PRINCIPLES), len(STATUSES))
            node_principle_logits[ntype] = logits

        present = [x_dict[nt] for nt in self.node_types if nt in x_dict]
        if not present:
            graph_score = torch.zeros(1)
        else:
            all_h = torch.cat(present, dim=0)
            mean_pool = all_h.mean(dim=0)
            max_pool = all_h.max(dim=0).values
            graph_score = self.score_head(
                torch.cat([mean_pool, max_pool], dim=-1)
            )

        return {
            "node_principle_logits": node_principle_logits,
            "graph_score": graph_score,
        }


__all__ = [
    "HeteroGAT",
    "HeteroGATConfig",
    "default_metadata",
]
