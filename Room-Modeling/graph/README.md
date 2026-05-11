# graph/ — Deterministic 3D scene graph builder

Consumes the editable `scene.json` produced by [`spatial/`](Room-Modeling/spatial/README.md)
and emits a typed scene graph (nodes + geometric, functional, and fengshui
edges) plus a dense per-pair relation matrix and a list of fengshui
`principle_checks`. The principle checks are the supervision signal for the
[`network/`](Room-Modeling/network/README.md) GNN.

## Pipeline position

```
scene.json  ─►  graph.cli            ─►  scene_graph.json (+ edges.csv)
                graph.sync_cache     ─►  outs/graph_cache/<room>/scene_graph.json (batch)
```

## Install

```bash
pip install -r requirements.txt
# or, editable:
pip install -e .
```

## CLI

Run from the repo root with `PYTHONPATH=.`. There are two entry points:

### `python -m graph` — single scene → scene_graph

| Flag | Default | Description |
|------|---------|-------------|
| `--scene PATH` | required | Path to a `scene.json` produced by `spatial/`. |
| `--out PATH` | sibling `scene_graph.json` | Output path for the JSON graph. |
| `--csv PATH` | sibling `edges.csv` | Output path for the edges CSV companion. |
| `--no_csv` | off | Skip the CSV companion entirely. |

Example:

```bash
PYTHONPATH=. python -m graph \
  --scene outs/spatial_editor_outputs/my_room/scene.json \
  --out   outs/spatial_editor_outputs/my_room/scene_graph.json
```

Prints a one-line summary like:

```
Wrote: .../scene_graph.json
Summary: 64 nodes, 421 edges, 7 zones, 3 violations, 5 warnings
```

### `python -m graph.sync_cache` — batch builder + cache

Discovers SpatialLM exports under a scan root and mirrors a parallel
`scene_graph.json` cache. Skips rooms whose cache is already at least as new
as the source `scene.json`. Renders a tqdm bar; only failures and the final
summary print.

| Flag | Default | Description |
|------|---------|-------------|
| `--scan-root DIR` | `outs/spatial_editor_outputs` | Where to look for `scene.json` files. |
| `--cache-dir DIR` | `outs/graph_cache` | Where to mirror `<room>/scene_graph.json`. |
| `--force` | off | Rebuild even when the cache is newer than the source. |
| `--dry-run` | off | Print plan only; write nothing. |
| `--no-csv` | off | Skip the per-room `edges.csv`. |
| `--recursive` | off | Descend into nested directories instead of inspecting only immediate subdirectories. |
| `--no-manifest` | off | Skip writing `<cache-dir>/manifest.json`. |

Example:

```bash
PYTHONPATH=. python -m graph.sync_cache
PYTHONPATH=. python -m graph.sync_cache --force            # rebuild after editing rules
PYTHONPATH=. python -m graph.sync_cache --recursive --dry-run
```

After a successful run `<cache-dir>/manifest.json` records every room's
status (`built` / `skipped` / `failed`) with timestamps for an audit trail.

## Module layout

- `cli.py` / `__main__.py` — single-room entry point.
- `sync_cache.py` — batch entry point + library API
  (`discover_scenes`, `is_stale`, `sync_room`, `sync_cache`, `write_manifest`).
- `geometry.py` — room polygon, walkable area, wall normals, entry doors.
- `scene_graph.py` — node / edge typing + dense relation matrix.
- `functional.py` — zone detection (sleep, work, social, …) + focal element.
- `fengshui.py` — `evaluate_principles(...)` produces `PrincipleCheck` rows
  attached as `principle_*` edges. **This is the teacher signal** the GNN is
  distilled from.
- `io.py` — `load_scene_json`, `write_scene_graph_json`, `write_edges_csv`.

## Output schema

`scene_graph_v1` keys: `room`, `nodes`, `edges`, `zones`,
`dense_relation_matrix`, `principle_checks`, `summary`.

## Tests

```bash
PYTHONPATH=. python -m pytest graph/tests -q
```
