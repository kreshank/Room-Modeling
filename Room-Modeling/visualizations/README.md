# Room graph viewer

Static **2D plan** viewer for `scene_graph.json` from the graph pipeline, optional GNN **predict** JSON, and an optional **extras** file for future LLM or attention dumps.

## Open

Double-click `index.html` or open it from the browser. Use the file pickers to load:

1. **scene_graph.json** (required) — produced by `python -m graph.cli` from SpatialLM `scene.json`.
2. **predict JSON** (optional) — output of `python -m network.cli predict` (`principle_predictions`, `graph_score`).
3. **extras JSON** (optional) — output of `python -m network.cli annotate`. When present, the sidebar shows:
   - Overall score label and summary from the LLM.
   - Numbered recommendations list.
   - Clickable ranked violations table (click a row to select and highlight that node on the canvas).
   - Per-node explanations in the hover/click detail panel.

   Supported extras shapes (for custom / future use):
   - `{ "explanations": [ { "target", "principle?", "text" | "summary" }, ... ] }` — per-node text.
   - `{ "edge_attention": [ ... ] }` — placeholder message until attention export is wired.

Edge stroke “weight” uses **numeric fields on graph edges** (e.g. `distance_m`, `relative_angle_deg`). These are **rule-engine / geometry inputs**, not learned GAT attention, unless you add an export that reuses the same field names.

## Layout

- **Walkable** (light gray) and **room polygon** outline.
- **Objects** as filled footprints from `geometry.corners_xy` (walls/doors/windows use the same schema when present).
- **Edges** between node anchors; filter by edge `type` in the sidebar. By default, `principle_*` edges are hidden.
- **Nodes**: click or hover for id, geometry snippet, and per-principle predictions when predict JSON is loaded. Optional coloring by worst predicted status (violated, then weak, then good).

No build step and no server required (file pickers avoid `file://` fetch CORS).
