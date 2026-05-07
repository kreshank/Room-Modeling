"""End-to-end orchestration for transcript extraction artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .aggregate import aggregate
from .clean import clean_records
from .ingest import ingest_xlsx_dir
from .io import read_jsonl, write_jsonl
from .rule_extract import extract_matches


def run_pipeline(
    *,
    input_dir: str | Path,
    out_dir: str | Path,
    stage: str = "all",
) -> dict[str, Any]:
    base_out = Path(out_dir)
    base_out.mkdir(parents=True, exist_ok=True)

    raw_path = base_out / "raw.jsonl"
    clean_path = base_out / "clean.jsonl"
    rule_path = base_out / "rule_extracted.jsonl"
    final_path = base_out / "final.jsonl"

    rows: list[dict[str, Any]]

    if stage in ("all", "ingest"):
        rows, stats = ingest_xlsx_dir(input_dir)
        write_jsonl(raw_path, rows)
    else:
        rows = list(read_jsonl(raw_path))

    if stage in ("all", "clean"):
        rows = clean_records(rows)
        write_jsonl(clean_path, rows)
    elif stage not in ("all", "ingest"):
        rows = list(read_jsonl(clean_path))

    if stage in ("all", "rules"):
        rows = extract_matches(rows)
        write_jsonl(rule_path, rows)
        write_jsonl(final_path, rows)
    elif stage == "aggregate":
        rows = list(read_jsonl(rule_path))

    if stage in ("all", "aggregate"):
        return aggregate(rows, base_out)

    return {"docs_total": len(rows)}

