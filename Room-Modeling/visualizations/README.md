# Room graph viewer

Static **2D plan** viewer for `scene_graph.json` from the graph pipeline, optional GNN **predict** JSON, and an optional **extras** file for future LLM or attention dumps.

## Open

Double-click `index.html` or open it from the browser. Use the file pickers to load:

1. **scene_graph.json** (required) — produced by `python -m graph.cli` from SpatialLM `scene.json`.
2. **predict JSON** (optional) — output of `python -m network.cli predict` (`principle_predictions`, `graph_score`).
3. **extras JSON** (optional) — not used by the pipeline yet. Supported shapes:
   - `{ "explanations": [ { "target", "principle?", "text" | "summary" }, ... ] }` — lines appear in the detail panel when hovering nodes.
   - `{ "edge_attention": [ ... ] }` — placeholder message until you export attention from Python.

Edge stroke “weight” uses **numeric fields on graph edges** (e.g. `distance_m`, `relative_angle_deg`). These are **rule-engine / geometry inputs**, not learned GAT attention, unless you add an export that reuses the same field names.

## Layout

- **Walkable** (light gray) and **room polygon** outline.
- **Objects** as filled footprints from `geometry.corners_xy` (walls/doors/windows use the same schema when present).
- **Edges** between node anchors; filter by edge `type` in the sidebar. By default, `principle_*` edges are hidden.
- **Nodes**: click or hover for id, geometry snippet, and per-principle predictions when predict JSON is loaded. Optional coloring by worst predicted status (violated, then weak, then good).

No build step and no server required (file pickers avoid `file://` fetch CORS).
