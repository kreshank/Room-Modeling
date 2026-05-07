"""CLI for transcript extraction pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .pipeline import run_pipeline


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest dm_shorts xlsx transcripts and build weak-supervision dataset"
    )
    parser.add_argument(
        "--in",
        dest="input_dir",
        required=True,
        help="Directory containing dm_shorts_*.xlsx files",
    )
    parser.add_argument(
        "--out",
        dest="out_dir",
        default="../outs/transcripts",
        help="Output directory for JSONL/summary artifacts",
    )
    parser.add_argument(
        "--stage",
        default="all",
        choices=["all", "ingest", "clean", "rules", "aggregate"],
        help="Run one stage or all stages",
    )
    args = parser.parse_args(argv)

    summary = run_pipeline(
        input_dir=Path(args.input_dir).expanduser().resolve(),
        out_dir=Path(args.out_dir).expanduser().resolve(),
        stage=args.stage,
    )
    print("Done.")
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

