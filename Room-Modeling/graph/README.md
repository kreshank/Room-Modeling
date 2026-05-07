# room_graph

Deterministic 3D scene graph builder for SpatialLM room outputs.

## Pipeline position

```
.ply -> SpatialLM inference -> layout.txt -> scene.json -> room_graph -> scene_graph.json
```

`scene.json` is produced by `spatiallm_room_editor`. This package consumes it
and emits a typed scene graph (nodes + geometric, functional, and feng-shui
edges) plus a dense per-pair relation matrix.

## Install

```bash
pip install -r requirements.txt
# or, editable:
pip install -e .
```

## Usage

```bash
python -m room_graph \
  --scene ../outs/spatial_editor_outputs/my_room/scene.json \
  --out   ../outs/spatial_editor_outputs/my_room/scene_graph.json
```

This writes `scene_graph.json` and a sibling `edges.csv` for inspection.

## Output schema

See `scene_graph_v1` keys: `room`, `nodes`, `edges`, `zones`,
`dense_relation_matrix`, `principle_checks`, `summary`.
