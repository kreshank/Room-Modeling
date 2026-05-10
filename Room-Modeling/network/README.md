# network/ — Feng shui GNN

A small heterogeneous GAT that ingests `scene_graph.json` (produced by the
`graph/` pipeline from a SpatialLM `scene.json`) and emits per-node
principle-contribution predictions plus a per-graph score.

This milestone delivers **architecture + I/O scaffolding only**. Training,
augmentation, and corpus tooling are deferred until labels arrive.

## Pipeline contract

```
.ply  ->  spatial/run_pipeline.py  ->  scene.json  ->  graph.cli  ->  scene_graph.json  ->  network/
```

## Install

CPU-only (works on AMD machines without ROCm):

```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
```

ROCm (AMD GPU):

```bash
pip install --index-url https://download.pytorch.org/whl/rocm6.1 torch>=2.3
pip install torch_geometric>=2.5 numpy>=1.24 networkx>=3.2
```

CUDA: the default `pip install -r requirements.txt` works.

## Quickstart (forward pass with random init)

```bash
python -m network.cli predict \
  --scene_graph ../outs/spatial_editor_outputs/smoke_test/scene_graph.json
```

Output (truncated):

```json
{
  "schema_version": "fengshui_gnn_v1",
  "graph_summary": { "nodes_per_type": { "object": 3, "wall": 4, ... } },
  "model_params": 138426,
  "graph_score": 0.4436,
  "principle_predictions": [
    {
      "principle": "command_position",
      "target": "bbox_2",
      "status": "weak",
      "score": 0.391
    }
  ]
}
```

The output mirrors the rule engine's `principle_checks` shape (`principle`,
`target`, `status`, `score`) so the GNN can later be substituted for
`evaluate_principles` downstream.

## Module layout

- `labels.py` — principle vocabulary, status enum, label embedding table,
  fine-grained edge-kind vocabulary (one-hot in `edge_attr`), per-node-type
  applicability mask.
- `data.py` — `scene_graph.json` → PyG `HeteroData`. Collapses the dozens of
  scene-graph edge type strings into ~10 `(src_kind, bucket, dst_kind)`
  relations; original type names are preserved as a one-hot tail on
  `edge_attr`. Filters `principle_*` edges. Applies `ToUndirected` so walls /
  rooms / zones receive reverse message paths without hand-written `_rev`
  suffixes.
- `model.py` — `HeteroGAT`: per-node-type input MLPs, 2 × `HeteroConv({rel:
  GATv2Conv(edge_dim=32)})` with residual + LayerNorm, per-node / per-graph
  heads. Relation keys mirror `data.py` / PyG's `rev_<bucket>` naming.
- `cli.py` — `predict` subcommand.

## What is intentionally not in this milestone

- Training loop, distillation loss, augmentation, evaluation against the rule
  engine.
- Any non-SpatialLM data adapter (ScanNet, etc.).
- Layout suggestion / placement head.
