"""Common JSONL/file helpers for tune pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return iter(())
    def _gen() -> Iterator[dict[str, Any]]:
        with p.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    return _gen()


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return p


def append_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return p
