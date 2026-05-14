from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUTS = ROOT / "outs"
DEFAULT_TRANSCRIPTS_DIR = OUTS / "transcripts"
DEFAULT_SPATIAL_ROOT = OUTS / "spatial_editor_outputs"
DEFAULT_GRAPH_CACHE = OUTS / "graph_cache"
DEFAULT_NETWORK_RUNS = OUTS / "network_runs"


def run_command(cmd: list[str]) -> None:
    print("\n>>> "+" ".join(map(str, cmd)))
    subprocess.run(cmd, cwd=ROOT, check=True)


def make_room_name(args: argparse.Namespace) -> str:
    if args.room_name:
        return args.room_name
    if args.ply:
        return Path(args.ply).stem
    if args.layout_txt:
        return Path(args.layout_txt).stem
    raise ValueError("Unable to infer room name without --room-name, --ply, or --layout-txt")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the full Room-Modeling pipeline end to end."
    )
    parser.add_argument("--ply", help="Input .ply point cloud for spatial parsing.")
    parser.add_argument(
        "--spatiallm-dir",
        help="Path to the cloned SpatialLM repository. Required unless --layout-txt is provided.",
    )
    parser.add_argument(
        "--layout-txt",
        help="Existing SpatialLM layout.txt to parse instead of running SpatialLM.",
    )
    parser.add_argument("--room-name", help="Name for the generated room output directory.")
    parser.add_argument(
        "--out-dir",
        help="Root output directory for spatial exports. Defaults to outs/spatial_editor_outputs/<room>.",
    )
    parser.add_argument("--skip-transcripts", action="store_true", help="Skip transcript summary generation.")
    parser.add_argument("--skip-spatial", action="store_true", help="Skip the spatial / scene.json generation step.")
    parser.add_argument("--skip-graph", action="store_true", help="Skip graph cache generation.")
    parser.add_argument("--skip-train", action="store_true", help="Skip model training.")
    parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation.")
    parser.add_argument("--skip-predict", action="store_true", help="Skip prediction.")
    parser.add_argument(
        "--summary-json",
        help="Optional summary.json for transcript-weighted training. If omitted, transcript step uses outs/transcripts/summary.json.",
    )
    parser.add_argument(
        "--checkpoint",
        help="Optional checkpoint (.pt) to evaluate / predict. If omitted, training produces a new checkpoint and best.pt is used.",
    )
    parser.add_argument(
        "--train-epochs",
        type=int,
        default=20,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--device",
        help="Device to use for training/evaluation (cuda, cpu, etc.).",
    )
    parser.add_argument(
        "--detect-type",
        default="all",
        choices=["all", "arch", "object"],
        help="SpatialLM detect type for run_pipeline.",
    )
    parser.add_argument(
        "--category",
        nargs="*",
        default=None,
        help="Optional category whitelist for SpatialLM.",
    )
    parser.add_argument(
        "--graph-scan-root",
        default=str(DEFAULT_SPATIAL_ROOT),
        help="Root scan path for graph.sync_cache.",
    )
    parser.add_argument(
        "--graph-cache-dir",
        default=str(DEFAULT_GRAPH_CACHE),
        help="Cache directory for generated scene_graph.json files.",
    )
    args = parser.parse_args(argv)

    if args.skip_spatial and args.skip_graph and args.skip_train and args.skip_eval and args.skip_predict:
        parser.error("At least one pipeline stage must run unless you want a no-op.")

    if not args.skip_spatial and args.layout_txt is None and args.ply is None:
        parser.error("--ply is required unless --skip-spatial or --layout-txt is supplied.")
    if not args.skip_spatial and args.layout_txt is None and not args.spatiallm_dir:
        parser.error("--spatiallm-dir is required unless --layout-txt is supplied.")

    room_name = make_room_name(args)
    spatial_out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else DEFAULT_SPATIAL_ROOT / room_name
    transcript_summary = Path(args.summary_json).expanduser().resolve() if args.summary_json else DEFAULT_TRANSCRIPTS_DIR / "summary.json"
    graph_cache_dir = Path(args.graph_cache_dir).expanduser().resolve()
    graph_glob = str(graph_cache_dir / "**" / "scene_graph.json")
    checkpoint_dir = DEFAULT_NETWORK_RUNS / f"train_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_transcripts:
        run_command(
            [
                sys.executable,
                "-m",
                "tune",
                "--in",
                str(Path("data/cliff_transcripts").resolve()),
                "--out",
                str(DEFAULT_TRANSCRIPTS_DIR.resolve()),
                "--stage",
                "all",
            ]
        )
        transcript_summary = DEFAULT_TRANSCRIPTS_DIR / "summary.json"

    if not args.skip_spatial:
        spatial_cmd = [
            sys.executable,
            "-m",
            "spatial.run_pipeline",
            "--out_dir",
            str(spatial_out_dir),
            "--detect_type",
            args.detect_type,
        ]
        if args.ply:
            spatial_cmd += ["--ply", str(Path(args.ply).expanduser().resolve())]
        if args.layout_txt:
            spatial_cmd += ["--layout_txt", str(Path(args.layout_txt).expanduser().resolve())]
        if args.spatiallm_dir:
            spatial_cmd += ["--spatiallm_dir", str(Path(args.spatiallm_dir).expanduser().resolve())]
        if args.category:
            spatial_cmd += ["--category"] + args.category
        run_command(spatial_cmd)

    if not args.skip_graph:
        run_command(
            [
                sys.executable,
                "-m",
                "graph.sync_cache",
                "--scan-root",
                str(Path(args.graph_scan_root).expanduser().resolve()),
                "--cache-dir",
                str(graph_cache_dir),
            ]
        )

    if not args.skip_train:
        train_cmd = [
            sys.executable,
            "-m",
            "network.cli",
            "train",
            "--train-glob",
            graph_glob,
            "--val-glob",
            graph_glob,
            "--epochs",
            str(args.train_epochs),
            "--checkpoint-dir",
            str(checkpoint_dir),
        ]
        if args.device:
            train_cmd += ["--device", args.device]
        if transcript_summary.exists():
            train_cmd += ["--summary-json", str(transcript_summary)]
        run_command(train_cmd)

    if not args.skip_eval:
        checkpoint = args.checkpoint or str(checkpoint_dir / "best.pt")
        run_command(
            [
                sys.executable,
                "-m",
                "network.cli",
                "eval",
                "--checkpoint",
                checkpoint,
                "--glob",
                graph_glob,
                "--out",
                str(DEFAULT_NETWORK_RUNS / f"eval_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}" / "metrics.json"),
            ]
        )

    if not args.skip_predict:
        checkpoint = args.checkpoint or str(checkpoint_dir / "best.pt")
        scene_graph_path = graph_cache_dir / room_name / "scene_graph.json"
        if not scene_graph_path.exists():
            raise FileNotFoundError(
                f"Scene graph not found for prediction: {scene_graph_path}. "
                "Run graph sync_cache first or adjust --graph-cache-dir / --room-name."
            )
        run_command(
            [
                sys.executable,
                "-m",
                "network.cli",
                "predict",
                "--scene_graph",
                str(scene_graph_path),
                "--weights",
                checkpoint,
            ]
        )

    print("\nFull pipeline completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
