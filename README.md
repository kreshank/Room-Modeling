# ScanNet Top-Down Layout Exporter + Editor

This toolchain converts one annotated ScanNet scene into:

- `scene_layout.json`: a labeled 2D scene graph for drag/drop editing
- `scene_layout.png`: a preview image
- `room_editor.html`: a lightweight browser editor for moving furniture in 2D

It is meant for **ScanNet scenes that already have annotation files**. That means it works on the downloaded ScanNet scene folders like:

- `scene0000_00_vh_clean_2.ply`
- `scene0000_00.aggregation.json`
- `scene0000_00_vh_clean_2.0.010000.segs.json`
- `scene0000_00.txt`
- `scene0000_00_vh_clean_2.labels.ply`
- `scannetv2-labels.combined.tsv`

This is the practical MVP path for your project:
1. Use ScanNet annotations to recover objects and labels.
2. Convert them into draggable 2D rectangles.
3. Build and test the UI on reliable ground-truth scenes first.
4. Later, replace the ScanNet annotation step with a prediction model for user `.ply` scans.

## Install

```bash
pip install -r requirements.txt
```

## Run

Example command:

```bash
python export_scannet_topdown.py \
  --scene_dir data/scannet/scans/scene0000_00 \
  --label_map data/scannet/scannetv2-labels.combined.tsv \
  --out_dir out/scene0000_00
```

Optional flags:

```bash
--include_non_movable        # keep structural items too
--min_points_per_object 20   # keep smaller instances
--room_resolution_m 0.04     # finer room polygon resolution
--no_png                     # JSON only
```

## Open the editor

Open `room_editor.html` in a browser, load the exported `scene_layout.json`, and drag the furniture.

### Controls

- Click object: select
- Drag object: move
- `Q` / `E`: rotate selected object
- Mouse wheel: zoom
- Hold `Space` + drag: pan
- `Export edited JSON`: save your changed layout

## Output JSON shape

The exported JSON looks like this:

```json
{
  "scene_id": "scene0000_00",
  "units": "meters",
  "room_polygon": [[...], [...]],
  "room_bbox": {
    "min_x": 0.0,
    "min_y": 0.0,
    "max_x": 4.3,
    "max_y": 3.8,
    "width": 4.3,
    "height": 3.8
  },
  "objects": [
    {
      "id": 1,
      "label": "bed",
      "cx": 1.7,
      "cy": 2.3,
      "width": 2.0,
      "depth": 1.5,
      "theta": 1.57,
      "z_min": 0.0,
      "z_max": 0.65
    }
  ]
}
```

## How the backend works

### 1. Mesh + alignment
- Reads vertex positions from `*_vh_clean_2.ply`
- Reads `axisAlignment` from `<scene>.txt`
- Applies the rigid transform so the scene is aligned to the room axes

### 2. Instance recovery
- Reads `segIndices` from `*_segs.json`
- Reads `segGroups` from `*.aggregation.json`
- Reconstructs each object instance by collecting all vertices in its segments

### 3. Room footprint
- Reads semantic labels from `*_vh_clean_2.labels.ply`
- Finds floor vertices
- Projects them onto XY
- Builds a room polygon using an occupancy-grid union approach

### 4. Draggable furniture rectangles
- For each object instance, computes a 2D oriented bounding box via PCA
- Exports center, size, angle, height range, and rectangle corners

## Important limitation

This implementation is **for annotated ScanNet scenes**. It does **not** solve generic user `.ply` furniture segmentation by itself.

For arbitrary user scans, you still need the extra model stage:

- floor / wall estimation
- 3D semantic segmentation
- 3D instance segmentation
- oriented box fitting

But the exported JSON/editor format is designed so that later prediction outputs can plug into the exact same UI.

## User `.ply` exporter default cleanup update

Use the existing v5 exporter for raw user scans:

```bash
python export_user_ply_topdown_v5.py \
  --ply data/user_scans/room_a.ply \
  --out_dir out/my_room_v5 \
  --scene_id my_room
```

The current v5 defaults are intentionally more conservative than the previous high-recall pass:

- `--v5_high_recall` is now **off by default** so the first run produces fewer duplicate tiny fragments.
- Furniture boxes snap to the nearest `0 / 90 / 180 / 270` degree orientation by default with `--snap_cardinal_angles`.
- Duplicate/fragment suppression is on by default with `--suppress_duplicate_objects`.
- Door/window detection still uses geometry-only evidence, but now adds editable `door_candidate` and `window_candidate` proxies when clean openings are not detected.

Useful flags:

```bash
# Re-enable the old high-recall behavior if the exporter misses too much furniture.
--v5_high_recall

# Disable cardinal snapping if the scan coordinate frame is rotated relative to the room.
--no-snap_cardinal_angles

# Keep more small objects if suppression removes too much.
--no-suppress_duplicate_objects

# Require more/fewer architectural fallback candidates.
--arch_min_door_candidates 1
--arch_min_window_candidates 2
```

The geometry-only exporter should still be treated as a proposal generator. For reliable labels such as `bed`, `sofa`, `desk`, `window`, and `door`, the stronger path is `export_user_ply_model_topdown.py` with predictions from a trained 3D segmentation model.

## User `.ply` exporter notes

The recommended raw user-scan command is:

```bash
python export_user_ply_topdown_v5.py \
  --ply data/user_scans/room_a.ply \
  --out_dir out/my_room_v5 \
  --scene_id my_room
```

The current `export_user_ply_topdown_v5.py` is self-contained with `user_ply_geometry_common.py`; it does not import earlier `export_user_ply_topdown_v#.py` files.

Recent v5 updates:

- Default output now snaps furniture orientations to the nearest 0/90/180/270 degrees.
- Orientation snapping now recomputes width/depth to enclose the original footprint, reducing wall clipping.
- Small duplicated fragments are merged into larger furniture proposals before final filtering.
- Floor-lamp detection is stricter, so fewer random fragments become `floor_lamp_or_tall_thin_object`.
- Bed/sofa/table/desk/counter classification now runs before cabinet/shelf classification, even when clutter raises `z_max`.
- Door/window proxy detection is more aggressive by default:
  - door proxies span from floor to about 2.1 m
  - window proxies span from about 0.75 m to 1.85 m
  - fallback proxies are labeled as `door` / `window` by default instead of `door_candidate` / `window_candidate`

Useful flags:

```bash
# fewer tiny detections
--min_object_points 25 --min_box_area 0.025 --max_objects 45

# disable fragment merging if it over-merges unrelated objects
--no-merge_fragments

# make architecture fallbacks explicit candidates instead of actual labels
--arch_fallback_as_candidate

# preserve original object angles instead of snapping to cardinal angles
--no-snap_cardinal_angles
```

Important limitation: this remains geometry-only detection. It can produce better proposals from a `.ply`, but reliable recognition of non-stereotypical furniture still requires a trained 3D segmentation or 2D/3D multi-view labeling model.
