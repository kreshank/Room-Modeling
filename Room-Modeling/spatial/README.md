# SpatialLM Room Editor Pipeline

This is a small implementation scaffold for:

```text
.ply point cloud
  -> official SpatialLM inference.py
  -> raw SpatialLM layout.txt
  -> editable scene.json + entities.csv
  -> browser-based top-down review editor
  -> user-confirmed JSON/CSV export
```

The project does **not** vendor SpatialLM. It calls the official SpatialLM repository as a subprocess, then parses the model's text output into an easier JSON schema for your own UI and layout heuristics.

## 1. Install the official SpatialLM repo

Follow the SpatialLM installation instructions in a Python 3.11/CUDA environment. Example:

```bash
git clone https://github.com/manycore-research/SpatialLM.git
cd SpatialLM
conda create -n spatiallm python=3.11
conda activate spatiallm
conda install -y -c nvidia/label/cuda-12.4.0 cuda-toolkit conda-forge::sparsehash
pip install poetry && poetry config virtualenvs.create false --local
poetry install
poe install-sonata
```

SpatialLM1.1 uses the Sonata point-cloud encoder, so `poe install-sonata` is the relevant extra install step for `manycore-research/SpatialLM1.1-Qwen-0.5B`.

## 2. Install this wrapper

From this folder:

```bash
pip install -r requirements.txt
```

If you want editable local imports while developing:

```bash
pip install -e .
```

The included `run_pipeline.py` works without packaging installation as long as you run it from this folder.

## 3. Run on an existing `.ply` file

From this folder, while using the same environment that can run SpatialLM:

```bash
python run_pipeline.py \
  --ply path/to/my_room.ply \
  --spatiallm_dir path/to/SpatialLM \
  --out_dir ../outs/spatial_editor_outputs/my_room \
  --model_path manycore-research/SpatialLM1.1-Qwen-0.5B \
  --detect_type all
```

Outputs:

```text
../outs/spatial_editor_outputs/my_room/layout.txt       # raw SpatialLM text output
../outs/spatial_editor_outputs/my_room/scene.json       # editable room scene schema
../outs/spatial_editor_outputs/my_room/entities.csv     # simple table export
../outs/spatial_editor_outputs/my_room/viewer/index.html
```

## 4. Use user-specified object categories

SpatialLM1.1 supports category-conditioned object detection. The wrapper defaults to a practical room/furniture set. You can override it:

```bash
python run_pipeline.py \
  --ply path/to/my_room.ply \
  --spatiallm_dir path/to/SpatialLM \
  --out_dir ../outs/spatial_editor_outputs/my_room \
  --detect_type all \
  --category bed desk chair sofa coffee_table tv_cabinet bookcase floor-standing_lamp curtain carpet
```

This is useful when you want fewer generic detections and more project-relevant categories.

## 5. Parse an existing SpatialLM layout file without rerunning the model

```bash
python run_pipeline.py \
  --ply path/to/my_room.ply \
  --layout_txt examples/sample_layout.txt \
  --out_dir ../outs/spatial_editor_outputs/sample_parse_only
```

## 6. Open the review UI

The easiest reliable option is to serve the output directory:

```bash
cd ../outs/spatial_editor_outputs/my_room
python -m http.server 8000
```

Then open:

```text
http://localhost:8000/viewer/index.html
```

The viewer tries to load `../scene.json` automatically. You can also use the file picker and upload any generated `scene.json`.

## 7. What the editor supports

- Top-down view using `x/y` coordinates and `z` as up.
- Walls, doors, windows, and furniture/object boxes.
- Dragging objects to adjust `x/y`.
- Numeric editing of position, size, height, kind, label, and yaw.
- Snapping selected object orientation to 0/90/180/270 degrees.
- Marking objects as confirmed.
- Adding manual objects.
- Deleting bad detections.
- Exporting a confirmed JSON or CSV file.

## 8. Scene JSON schema

Each object becomes an editable entity:

```json
{
  "id": "bbox_0",
  "kind": "furniture",
  "label": "desk",
  "x": 2.1,
  "y": 1.4,
  "z": 0.4,
  "yaw_rad": 1.57,
  "yaw_deg": 90.0,
  "width": 1.2,
  "depth": 0.7,
  "height": 0.8,
  "confirmed": false,
  "source": "spatiallm",
  "raw": {}
}
```

Walls are converted from two endpoints into a top-down rectangle with center, length, thickness, height, and yaw. Doors/windows inherit orientation from their referenced wall when possible.

## 9. Important assumptions

- The `.ply` file should be a point cloud, not just a mesh-only file.
- The point cloud should be metric-scale where 1 unit is approximately 1 meter.
- The point cloud should be z-up and preferably roughly axis-aligned to the room.
- If your scanner exports a tilted or arbitrary-coordinate point cloud, align/clean it before inference.

## 10. Recommended next integration step

Use the exported `room_scene_confirmed.json` as the input to your room-layout scoring pipeline. For example:

```text
confirmed JSON
  -> build occupancy grid
  -> detect walkable floor area
  -> compute door visibility / command position
  -> compute window/light proximity
  -> produce recommendations + editable layout suggestions
```
