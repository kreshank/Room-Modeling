# network/ — Feng shui GNN

A small heterogeneous GAT that ingests `scene_graph.json` (produced by the
[`graph/`](Room-Modeling/graph/README.md) pipeline) and emits per-node
principle-contribution predictions plus a per-graph fengshui score.

The model is trained by **distilling** the deterministic rule engine in
[`graph/fengshui.py`](Room-Modeling/graph/fengshui.py): each scene graph
already contains `principle_checks` emitted by `evaluate_principles`, and
those rows are the supervision signal. No human labels, no transcripts as
direct targets — [`tune/`](Room-Modeling/tune/README.md) summary counts
optionally serve as per-principle loss weights.

## Pipeline contract

```
.ply  ─►  spatial/run_pipeline.py  ─►  scene.json
                                            │
                                            ▼
                              graph.sync_cache (or graph.cli)
                                            │
                                            ▼
                                    scene_graph.json
                                            │
                                            ▼
                                  network.cli {predict,train,eval}
```

## Install

CPU-only (works on AMD machines without ROCm):

```bash
pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cpu
```

ROCm (AMD GPU):

```bash
pip install --index-url https://download.pytorch.org/whl/rocm6.1 torch>=2.3
pip install torch_geometric>=2.5 numpy>=1.24 networkx>=3.2 tqdm>=4.65
```

CUDA: the default `pip install -r requirements.txt` works.

## CLI

Run from the repo root with `PYTHONPATH=.`. Three subcommands share one
entry point: `python -m network.cli {predict,train,eval}`.

Default JSON output: `predict` → `outs/inference/...`; `train` / `eval` →
`outs/network_runs/...`. Summaries print to stdout; training shows tqdm bars.
`--device` defaults to `cuda` when available and falls back to `cpu`.

### `python -m network.cli predict`

Run a forward pass on one scene graph.

| Flag | Default | Description |
|------|---------|-------------|
| `--scene_graph PATH` | required | `scene_graph.json` produced by `graph/`. |
| `--weights PATH` | none (random init) | `.pt` checkpoint to load. |
| `--out PATH` | `outs/inference/predict_<scene>_<timestamp>.json` | Where to write the predictions JSON. |
| `--seed INT` | `0` | Torch seed. |

Example:

```bash
PYTHONPATH=. python -m network.cli predict \
  --scene_graph outs/graph_cache/my_room/scene_graph.json \
  --weights     outs/network_runs/train_2026-05-10_20-16-51/best.pt
```

Prints a compact summary (source, weights, params, `graph_score`,
nodes-per-type, predictions broken down by status). The full per-cell
predictions go to the JSON file.

### `python -m network.cli train`

Distillation training against `principle_checks` taken from one or more
cached `scene_graph.json` files.

| Flag | Default | Description |
|------|---------|-------------|
| `--train-glob PAT` (repeatable) | — | Glob for training graphs. |
| `--val-glob PAT` (repeatable) | — | Glob for validation graphs. |
| `--train-manifest PATH` | — | Newline-separated paths or `.jsonl` with a `scene_graph` key. |
| `--val-manifest PATH` | — | Same as above for the validation set. |
| `--epochs INT` | `20` | Number of epochs. |
| `--lr FLOAT` | `1e-3` | AdamW learning rate. |
| `--weight-decay FLOAT` | `1e-4` | AdamW weight decay. |
| `--lambda-score FLOAT` | `0.5` | Weight on per-cell expected-score smooth-L1. |
| `--lambda-graph FLOAT` | `0.25` | Weight on per-graph score smooth-L1. |
| `--summary-json PATH` | — | `outs/transcripts/summary.json` for `log(1+count)` per-principle CE weights. Uniform when omitted. |
| `--checkpoint-dir DIR` | `outs/network_runs/train_<timestamp>/` | Output directory for `best.pt`, `last.pt`, `log.jsonl`, `history.json`. |
| `--device STR` | `cuda` if available else `cpu` | `cuda`, `cpu`, `cuda:1`, etc. |
| `--seed INT` | `0` | Torch / shuffle seed. |

Example:

```bash
PYTHONPATH=. python -m network.cli train \
  --train-glob 'outs/graph_cache/**/scene_graph.json' \
  --val-glob   'outs/graph_cache/**/scene_graph.json' \
  --epochs 20 \
  --summary-json outs/transcripts/summary.json
```

Per-epoch you'll see one summary line:

```
epoch  3/20 | loss=0.842 | val_f1=0.413 | score_mae=0.21 | graph_mae=0.07  *best (saved)
```

Plus tqdm bars during training and validation. Files written to
`--checkpoint-dir`:

| File | Contents |
|------|----------|
| `best.pt` | Best `val.macro_f1` checkpoint. |
| `last.pt` | Final-epoch checkpoint. |
| `log.jsonl` | One JSON object per epoch (loss, components, val metrics, checkpoint path). |
| `history.json` | Aggregate (`history`, `best_macro_f1`, `best_checkpoint`). |

### `python -m network.cli eval`

Score a checkpoint against the rule engine on a held-out set.

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint PATH` | — (random init) | `.pt` to load. |
| `--glob PAT` (repeatable) | — | Glob for evaluation graphs. |
| `--manifest PATH` | — | Newline-separated paths or `.jsonl`. |
| `--device STR` | `cuda` if available else `cpu` | Same semantics as `train`. |
| `--out PATH` | `outs/network_runs/eval_<timestamp>/metrics.json` | Where to write metrics JSON. |

Example:

```bash
PYTHONPATH=. python -m network.cli eval \
  --checkpoint outs/network_runs/train_2026-05-10_20-16-51/best.pt \
  --glob       'outs/graph_cache/**/scene_graph.json'
```

Prints a tabular summary (macro-F1, score MAE, graph score MAE, supervised
cell counts, per-principle F1) and writes the full metrics JSON.

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
  GATv2Conv(edge_dim=32)})` with residual + LayerNorm, per-node and
  per-graph heads. Relation keys mirror `data.py` / PyG's `rev_<bucket>`
  naming. ~138k params at default config.
- `targets.py` — aligns each `principle_check` row with the `(node_type,
  node_idx, principle_idx)` cell the model emits. Unsupervised cells are
  filled with `IGNORE_INDEX` (`-1`) so a vanilla `cross_entropy` skips them.
- `dataset.py` — `FengShuiSceneGraphDataset(paths)` returning `(HeteroData,
  targets, id_order, path)`. `resolve_paths(globs=, manifest=)` powers the
  CLI input flags.
- `metrics.py` — pure-torch per-principle macro-F1 on status (violated /
  weak / good) and masked MAE for scores.
- `train.py` — distillation loop. Loss = `weighted CE(status)` +
  `λ_score · SmoothL1(expected_score, teacher_score)` +
  `λ_graph · SmoothL1(σ(graph_logit), mean teacher score)`. Tqdm bars,
  JSONL log, human-readable summary printer.
- `eval.py` — load checkpoint, run held-out set, emit metrics JSON.
- `cli.py` / `__main__.py` — `predict`, `train`, `eval` subcommands.

## Tests

```bash
PYTHONPATH=. python -m pytest network/tests -q
```

## What is still deferred

- Heterogeneous-graph batching (currently effective `batch_size=1`).
- Jitter / augmentation that re-runs `graph.cli` on perturbed scenes for
  more training data.
- ScanNet → SpatialLM bulk export pipeline.
- Layout-suggestion / placement head.
