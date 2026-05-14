import argparse
import subprocess
from pathlib import Path


def run(cmd, cwd=None):
    print("\nRunning:")
    print(" ".join(str(x) for x in cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ply", required=True)
    parser.add_argument("--spatiallm_dir", required=True)
    parser.add_argument("--principles_xlsx", required=True)
    parser.add_argument("--out_dir", required=True)

    args = parser.parse_args()

    root = Path(__file__).parent.resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    spatial_out = out_dir / "spatial"
    graph_out = out_dir / "graph"
    network_out = out_dir / "network"
    tune_out = out_dir / "tune"
    final_out = out_dir / "final"

    spatial_out.mkdir(exist_ok=True)
    graph_out.mkdir(exist_ok=True)
    network_out.mkdir(exist_ok=True)
    tune_out.mkdir(exist_ok=True)
    final_out.mkdir(exist_ok=True)

    # 1. SpatialLM: .ply -> scene.json
    run([
        "python", "run_pipeline.py",
        "--ply", args.ply,
        "--spatiallm_dir", args.spatiallm_dir,
        "--out_dir", str(spatial_out),
        "--model_path", "manycore-research/SpatialLM1.1-Qwen-0.5B",
        "--detect_type", "all"
    ], cwd=root / "spatial")

    scene_json = spatial_out / "scene.json"

    # 2. Graph pipeline: scene.json -> graph.json
    run([
        "python", "graph_cli.py",
        "--scene_json", str(scene_json),
        "--out_dir", str(graph_out)
    ], cwd=root / "graph")

    graph_json = graph_out / "graph.json"

    # 3. Tune pipeline: xlsx principles -> principles.json
    run([
        "python", "graph_cli.py",
        "--xlsx", args.principles_xlsx,
        "--out_dir", str(tune_out)
    ], cwd=root / "tune")

    principles_json = tune_out / "principles.json"

    # 4. Network/GNN: graph.json -> ratings.json
    run([
        "python", "run_gnn.py",
        "--graph_json", str(graph_json),
        "--principles_json", str(principles_json),
        "--out_dir", str(network_out)
    ], cwd=root / "network")

    ratings_json = network_out / "ratings.json"

    # 5. LLM: ratings + principles + graph -> final recommendation
    run([
        "python", "recommend.py",
        "--scene_json", str(scene_json),
        "--graph_json", str(graph_json),
        "--ratings_json", str(ratings_json),
        "--principles_json", str(principles_json),
        "--out_dir", str(final_out)
    ], cwd=root / "llm")

    print("\nPipeline complete!")
    print(f"Final output: {final_out / 'final_report.txt'}")


if __name__ == "__main__":
    main()