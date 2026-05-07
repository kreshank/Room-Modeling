"""Aggregate extracted records into dataset + summary artifacts."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .io import write_jsonl


def aggregate(rows: list[dict[str, Any]], out_dir: str | Path) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    dataset_rows = [r for r in rows if r.get("topic", {}).get("is_on_topic", False)]
    write_jsonl(out / "dataset.jsonl", dataset_rows)

    by_year = Counter(str(r.get("year", "")) for r in rows)
    on_topic = sum(1 for r in rows if r.get("topic", {}).get("is_on_topic", False))
    principle_counts = Counter()
    action_counts = Counter()
    target_counts = Counter()
    by_year_principle: dict[str, Counter[str]] = defaultdict(Counter)
    examples: dict[str, list[tuple[float, str]]] = defaultdict(list)

    for r in dataset_rows:
        year = str(r.get("year", ""))
        for m in r.get("principles", []):
            p = str(m.get("principle", "unknown"))
            principle_counts[p] += 1
            by_year_principle[year][p] += 1
            a = m.get("action")
            t = m.get("target")
            if a:
                action_counts[str(a)] += 1
            if t:
                target_counts[str(t)] += 1
            span = str(m.get("evidence_span", "")).strip()
            conf = float(m.get("confidence", 0.0))
            if span:
                examples[p].append((conf, span))

    summary = {
        "docs_total": len(rows),
        "docs_on_topic": on_topic,
        "docs_off_topic": len(rows) - on_topic,
        "docs_by_year": dict(by_year),
        "principle_counts": dict(principle_counts),
        "action_counts": dict(action_counts),
        "target_counts": dict(target_counts),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    years = sorted(by_year_principle.keys())
    principles = sorted({p for c in by_year_principle.values() for p in c.keys()})
    with (out / "principle_freq.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["principle", *years])
        for p in principles:
            writer.writerow([p, *[by_year_principle[y].get(p, 0) for y in years]])

    ex_dir = out / "examples_by_principle"
    ex_dir.mkdir(parents=True, exist_ok=True)
    for principle, spans in examples.items():
        spans_sorted = sorted(spans, key=lambda x: x[0], reverse=True)
        seen_text: set[str] = set()
        top: list[dict[str, Any]] = []
        for conf, span in spans_sorted:
            if span in seen_text:
                continue
            seen_text.add(span)
            top.append({"confidence": conf, "evidence_span": span})
            if len(top) >= 10:
                break
        (ex_dir / f"{principle}.json").write_text(
            json.dumps(top, indent=2), encoding="utf-8"
        )

    return summary

