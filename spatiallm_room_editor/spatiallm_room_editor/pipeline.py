"""Run a .ply file through SpatialLM and export editable room JSON.

This wrapper assumes you have cloned the official SpatialLM repo separately.
It intentionally calls SpatialLM's own inference.py by subprocess, so you do
not have to modify their model code.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from .parser import parse_layout_file

DEFAULT_MODEL = "manycore-research/SpatialLM1.1-Qwen-0.5B"
DEFAULT_CATEGORIES = [
    "bed",
    "desk",
    "chair",
    "sofa",
    "coffee_table",
    "dining_table",
    "side_table",
    "nightstand",
    "wardrobe",
    "tv_cabinet",
    "bookcase",
    "sideboard",
    "cupboard",
    "dressing_table",
    "stool",
    "carpet",
    "curtain",
    "plants",
    "tv",
    "computer",
    "floor-standing_lamp",
    "chandelier",
    "painting",
    "mirror",
]


def run_spatiallm_inference(
    *,
    ply_path: Path,
    layout_txt_path: Path,
    spatiallm_dir: Path,
    model_path: str,
    detect_type: str = "all",
    categories: Sequence[str] | None = None,
    python_executable: str = sys.executable,
    no_cleanup: bool = False,
    extra_args: Sequence[str] | None = None,
) -> None:
    """Call official SpatialLM inference.py with a .ply input.

    For detect_type="all", categories can still be passed to condition object
    boxes while preserving architectural detection in current SpatialLM versions.
    """

    inference_py = spatiallm_dir / "inference.py"
    code_template = spatiallm_dir / "code_template.txt"
    if not inference_py.exists():
        raise FileNotFoundError(f"Could not find SpatialLM inference.py at {inference_py}")
    if not code_template.exists():
        raise FileNotFoundError(f"Could not find SpatialLM code_template.txt at {code_template}")
    if not ply_path.exists():
        raise FileNotFoundError(f"Could not find input .ply file at {ply_path}")

    layout_txt_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_executable,
        str(inference_py),
        "--point_cloud",
        str(ply_path.resolve()),
        "--output",
        str(layout_txt_path.resolve()),
        "--model_path",
        model_path,
        "--detect_type",
        detect_type,
        "--code_template_file",
        str(code_template.resolve()),
    ]
    if categories:
        cmd.extend(["--category", *categories])
    if no_cleanup:
        cmd.append("--no_cleanup")
    if extra_args:
        cmd.extend(extra_args)

    print("Running SpatialLM:")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=str(spatiallm_dir), check=True)


def write_entities_csv(scene: dict, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "kind",
        "label",
        "x",
        "y",
        "z",
        "yaw_rad",
        "yaw_deg",
        "width",
        "depth",
        "height",
        "confirmed",
        "notes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(scene.get("entities", []))


def copy_viewer(viewer_out_dir: Path) -> None:
    viewer_src = Path(__file__).resolve().parents[1] / "viewer"
    viewer_out_dir.mkdir(parents=True, exist_ok=True)
    for item in viewer_src.iterdir():
        dest = viewer_out_dir / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)


def build_outputs(
    *,
    ply_path: Path,
    layout_txt_path: Path,
    out_dir: Path,
    model_path: str,
    make_viewer: bool = True,
) -> dict:
    scene_json_path = out_dir / "scene.json"
    scene = parse_layout_file(
        layout_txt_path,
        scene_json_path,
        source_ply=str(ply_path),
        model_path=model_path,
    )
    write_entities_csv(scene, out_dir / "entities.csv")
    if make_viewer:
        copy_viewer(out_dir / "viewer")
        # Convenience copy: opening viewer/index.html can default-load ../scene.json.
        print(f"Viewer written to: {out_dir / 'viewer' / 'index.html'}")
    return scene


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=".ply → SpatialLM layout → editable room-scene JSON")
    parser.add_argument("--ply", required=True, help="Path to the input .ply point cloud")
    parser.add_argument("--out_dir", default="outputs/my_room", help="Directory for layout.txt, scene.json, CSV, and viewer")
    parser.add_argument("--spatiallm_dir", default=None, help="Path to cloned official SpatialLM repo")
    parser.add_argument("--model_path", default=DEFAULT_MODEL)
    parser.add_argument("--detect_type", default="all", choices=["all", "arch", "object"])
    parser.add_argument("--category", nargs="*", default=None, help="Optional SpatialLM object categories. Defaults to a room/furniture set.")
    parser.add_argument("--layout_txt", default=None, help="Use an existing SpatialLM layout txt instead of running inference")
    parser.add_argument("--python", default=sys.executable, help="Python executable inside your SpatialLM environment")
    parser.add_argument("--no_cleanup", action="store_true", help="Pass --no_cleanup to SpatialLM")
    parser.add_argument("--no_viewer", action="store_true", help="Do not copy the HTML editor into the output folder")
    parser.add_argument("--build_graph", action="store_true", help="Also build a deterministic scene_graph.json via the room_graph package")
    parser.add_argument("--extra_spatiallm_args", nargs=argparse.REMAINDER, help="Advanced args passed after -- to SpatialLM inference.py")
    args = parser.parse_args(argv)

    ply_path = Path(args.ply).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    layout_txt_path = Path(args.layout_txt).expanduser().resolve() if args.layout_txt else out_dir / "layout.txt"

    categories = args.category if args.category is not None else DEFAULT_CATEGORIES

    if args.layout_txt is None:
        if not args.spatiallm_dir:
            parser.error("--spatiallm_dir is required unless --layout_txt is provided")
        run_spatiallm_inference(
            ply_path=ply_path,
            layout_txt_path=layout_txt_path,
            spatiallm_dir=Path(args.spatiallm_dir).expanduser().resolve(),
            model_path=args.model_path,
            detect_type=args.detect_type,
            categories=categories,
            python_executable=args.python,
            no_cleanup=args.no_cleanup,
            extra_args=args.extra_spatiallm_args,
        )

    scene = build_outputs(
        ply_path=ply_path,
        layout_txt_path=layout_txt_path,
        out_dir=out_dir,
        model_path=args.model_path,
        make_viewer=not args.no_viewer,
    )

    print(f"Wrote: {out_dir / 'scene.json'}")
    print(f"Wrote: {out_dir / 'entities.csv'}")
    print(f"Parsed {len(scene.get('entities', []))} entities; skipped {len(scene.get('skipped_lines', []))} lines.")

    if args.build_graph:
        try:
            from room_graph.cli import build_from_scene_json
            from room_graph.io import write_edges_csv, write_scene_graph_json
        except ImportError as exc:
            print(
                "--build_graph requested but room_graph is not installed. "
                "Install it from the sibling package: pip install -e ../room_graph",
                file=sys.stderr,
            )
            print(f"Import error: {exc}", file=sys.stderr)
            return 0

        scene_graph = build_from_scene_json(out_dir / "scene.json")
        graph_path = out_dir / "scene_graph.json"
        write_scene_graph_json(scene_graph, graph_path)
        write_edges_csv(scene_graph, out_dir / "edges.csv")
        print(f"Wrote: {graph_path}")
        print(f"Wrote: {out_dir / 'edges.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
