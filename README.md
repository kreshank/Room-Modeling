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
