"""Dataset wrapper that pairs ``scene_graph.json`` files with teacher tensors."""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

from torch.utils.data import Dataset

from .data import load_scene_graph, to_hetero_data
from .targets import build_teacher_tensors


def _expand_globs(patterns: Sequence[str]) -> list[Path]:
    seen: dict[str, None] = {}
    for pat in patterns:
        for match in glob.glob(pat, recursive=True):
            if match.endswith(".json"):
                seen.setdefault(str(Path(match).resolve()), None)
    return [Path(p) for p in seen.keys()]


def _read_manifest(manifest_path: str | Path) -> list[Path]:
    out: list[Path] = []
    base = Path(manifest_path).resolve().parent
    text = Path(manifest_path).read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("{"):
            obj = json.loads(line)
            cand = obj.get("scene_graph") or obj.get("path")
            if not cand:
                continue
            p = Path(cand)
        else:
            p = Path(line)
        if not p.is_absolute():
            p = (base / p).resolve()
        out.append(p)
    return out


def resolve_paths(
    *,
    globs: Sequence[str] | None = None,
    manifest: str | Path | None = None,
) -> list[Path]:
    """Resolve a list of ``scene_graph.json`` paths from ``--glob`` / ``--manifest``."""

    paths: list[Path] = []
    if globs:
        paths.extend(_expand_globs(globs))
    if manifest:
        paths.extend(_read_manifest(manifest))
    deduped: dict[str, Path] = {}
    for p in paths:
        deduped.setdefault(str(p.resolve()), p)
    return list(deduped.values())


class FengShuiSceneGraphDataset(Dataset):
    """A list of ``scene_graph.json`` paths surfaced as ``(HeteroData, targets)``."""

    def __init__(self, paths: Iterable[str | Path]):
        self.paths: list[Path] = [Path(p) for p in paths]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[Any, dict[str, Any], dict[str, list[str]], Path]:
        path = self.paths[idx]
        scene_graph = load_scene_graph(path)
        data, id_order = to_hetero_data(scene_graph)
        targets = build_teacher_tensors(scene_graph, id_order)
        return data, targets, id_order, path


__all__ = [
    "FengShuiSceneGraphDataset",
    "resolve_paths",
]
