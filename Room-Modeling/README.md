# Room-Modeling

Point cloud → scene → rule-engine graph → GNN that distills the rules. Optional
transcript priors weight the training loss by how often people discuss each
principle.

```
.ply  ─►  spatial/   ─►  scene.json
                      │
                      ▼
                   graph/      ─►  scene_graph.json  (teacher)
                      │
                      ▼
                  network/     ─►  predictions  (student)

tune/  ─►  summary.json  (optional loss weights, not labels)
```

Work from this folder (`Room-Modeling/`). You normally do not need
`PYTHONPATH=` — `python -m …` puts the current directory on `sys.path` so
`graph`, `network`, and `tune` import cleanly. If you ever run from another
working directory and imports fail, set `PYTHONPATH` to this folder (or use
an editable install — see each package's `pyproject.toml`).

## Pipeline CLI

Steps run top-to-bottom. Omit optional steps if you already have the artifacts.

### 1. Transcripts (optional)

Build `outs/transcripts/summary.json` for `--summary-json` during training.
Skip if that file already exists.

```bash
python -m tune --in data/cliff_transcripts --out outs/transcripts
```

### 2. SpatialLM → scene.json

Needs a CUDA machine with SpatialLM installed. From [`spatial/`](spatial/README.md):

```bash
cd spatial
python run_pipeline.py --ply path/to/room.ply --spatiallm_dir path/to/SpatialLM
cd ..
```

Defaults: `--out_dir ../outs/spatial_editor_outputs/my_room`, `--detect_type all`.
Adds `scene.json`, `layout.txt`, `entities.csv`, and `viewer/` under that folder.

Re-use an existing `layout.txt` without GPU (parse only): pass
`--layout_txt path/to/layout.txt` and drop `--spatiallm_dir`.

### 3. scene.json → graph cache

Builds `outs/graph_cache/<room>/scene_graph.json` from every export under
`outs/spatial_editor_outputs/`. Skips rooms already up to date.

```bash
python -m graph.sync_cache
```

Defaults: `--scan-root outs/spatial_editor_outputs`, `--cache-dir outs/graph_cache`.

Single room instead of the cache:

```bash
python -m graph --scene outs/spatial_editor_outputs/my_room/scene.json
```

Writes `scene_graph.json` next to `scene.json` unless you pass `--out`.

### 4. Train

```bash
python -m network.cli train \
  --train-glob 'outs/graph_cache/**/scene_graph.json' \
  --val-glob   'outs/graph_cache/**/scene_graph.json' \
  --summary-json outs/transcripts/summary.json
```

Defaults: `--epochs 20`, `--device` = CUDA if available else CPU,
`--checkpoint-dir outs/network_runs/train_<timestamp>/`. Drop `--summary-json`
for uniform per-principle loss weights.

### 5. Evaluate

```bash
python -m network.cli eval \
  --checkpoint outs/network_runs/train_<timestamp>/best.pt \
  --glob 'outs/graph_cache/**/scene_graph.json'
```

Defaults: `--device` as above, `--out outs/network_runs/eval_<timestamp>/metrics.json`.

### 6. Predict one room

```bash
python -m network.cli predict \
  --scene_graph outs/graph_cache/my_room/scene_graph.json \
  --weights     outs/network_runs/train_<timestamp>/best.pt
```

Defaults JSON output under `outs/inference/predict_<scene>_<timestamp>.json`.

### 7. Full-pipeline script

Run the full end-to-end workflow from point cloud to prediction with the helper script:

```bash
python run_full_pipeline.py \
  --ply path/to/room.ply \
  --spatiallm-dir path/to/SpatialLM
```

Optional flags:
- `--skip-transcripts`
- `--skip-spatial`
- `--skip-graph`
- `--skip-train`
- `--skip-eval`
- `--skip-predict`

Use `--out-dir` to change the root output location, and `--checkpoint` to evaluate or predict with an existing model.

## Modules

| Folder | Role |
|--------|------|
| [`spatial/`](spatial/README.md) | `.ply` → SpatialLM → `scene.json` |
| [`graph/`](graph/README.md) | `scene.json` → `scene_graph.json` |
| [`network/`](network/README.md) | Train / eval / predict on graphs |
| [`tune/`](tune/README.md) | Transcripts → counts for loss weights |

## Outputs (`outs/`, gitignored)

```
outs/spatial_editor_outputs/<room>/   # scene.json, layout.txt, viewer, …
outs/graph_cache/<room>/              # scene_graph.json, edges.csv
outs/transcripts/                     # tune: summary.json, *.jsonl
outs/network_runs/                    # train / eval artifacts
outs/inference/                       # default predict JSON output
```

## Tests

```bash
python -m pytest graph/tests -q
python -m pytest network/tests -q
```
