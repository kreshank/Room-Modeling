"""Batch builder: sync ``scene.json`` exports into a ``scene_graph.json`` cache.

Discovers SpatialLM room exports under a scan root, builds the deterministic
scene graph for each room with :func:`graph.cli.build_from_scene_json`, and
mirrors the result into a cache directory. Re-runs are cheap: a room is only
rebuilt when its cached graph is missing or older than the source ``scene.json``
(or when ``--force`` is given). This is the producer side of the
``scan_root  ->  cache_dir  ->  network.cli train --train-glob`` workflow.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

from tqdm.auto import tqdm

from .cli import build_from_scene_json
from .io import write_edges_csv, write_scene_graph_json


DEFAULT_SCAN_ROOT = Path("outs/spatial_editor_outputs")
DEFAULT_CACHE_DIR = Path("outs/graph_cache")

STATUS_BUILT = "built"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"
STATUS_WOULD_BUILD = "would_build"
STATUS_WOULD_SKIP = "would_skip"


@dataclass
class SyncResult:
    """One row in the sync report — what we did with a single room."""

    room: str
    source: str
    destination: str
    status: str
    error: str | None = None


def discover_scenes(
    scan_root: Path, *, recursive: bool
) -> list[tuple[str, Path]]:
    """Return ``(room_key, scene_json_path)`` pairs sorted by ``room_key``.

    In default mode (``recursive=False``) each immediate subdirectory of
    ``scan_root`` that contains ``scene.json`` is treated as one room. In
    recursive mode every ``scene.json`` discovered under ``scan_root`` becomes
    a room, keyed by the relative parent path (e.g. ``foo/bar``) so the cache
    mirrors the source layout.
    """

    pairs: list[tuple[str, Path]] = []
    if not scan_root.exists():
        return pairs

    if recursive:
        for scene in scan_root.rglob("scene.json"):
            try:
                rel = scene.parent.relative_to(scan_root)
            except ValueError:
                continue
            if rel == Path("."):
                continue  # ignore a scene.json sitting at scan_root itself
            pairs.append((rel.as_posix(), scene))
    else:
        for sub in scan_root.iterdir():
            if not sub.is_dir():
                continue
            candidate = sub / "scene.json"
            if candidate.is_file():
                pairs.append((sub.name, candidate))

    pairs.sort(key=lambda pair: pair[0])
    return pairs


def is_stale(source: Path, dest: Path) -> bool:
    """``True`` when ``dest`` is missing or older than ``source``."""

    if not dest.exists():
        return True
    return dest.stat().st_mtime < source.stat().st_mtime


def sync_room(
    room: str,
    scene_path: Path,
    cache_dir: Path,
    *,
    force: bool,
    dry_run: bool,
    write_csv: bool,
) -> SyncResult:
    """Build (or skip) one room. Failures are captured, never raised."""

    dest = cache_dir / room / "scene_graph.json"
    rebuild = force or is_stale(scene_path, dest)

    if not rebuild:
        return SyncResult(
            room=room,
            source=str(scene_path),
            destination=str(dest),
            status=STATUS_WOULD_SKIP if dry_run else STATUS_SKIPPED,
        )

    if dry_run:
        return SyncResult(
            room=room,
            source=str(scene_path),
            destination=str(dest),
            status=STATUS_WOULD_BUILD,
        )

    try:
        scene_graph = build_from_scene_json(scene_path)
        write_scene_graph_json(scene_graph, dest)
        if write_csv:
            write_edges_csv(scene_graph, dest.with_name("edges.csv"))
    except Exception as exc:  # noqa: BLE001 — surface per-room, keep batch alive
        return SyncResult(
            room=room,
            source=str(scene_path),
            destination=str(dest),
            status=STATUS_FAILED,
            error=f"{type(exc).__name__}: {exc}",
        )

    return SyncResult(
        room=room,
        source=str(scene_path),
        destination=str(dest),
        status=STATUS_BUILT,
    )


def sync_cache(
    scan_root: str | Path = DEFAULT_SCAN_ROOT,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    *,
    force: bool = False,
    dry_run: bool = False,
    write_csv: bool = True,
    recursive: bool = False,
    show_progress: bool = False,
) -> list[SyncResult]:
    """Sync every ``scene.json`` under ``scan_root`` into ``cache_dir``.

    Pass ``show_progress=True`` (the CLI does this) to render a tqdm bar over
    rooms with the current room name in the postfix.
    """

    scan = Path(scan_root)
    cache = Path(cache_dir)
    pairs = discover_scenes(scan, recursive=recursive)
    iterator: Iterable[tuple[str, Path]] = pairs
    bar: tqdm | None = None
    if show_progress and pairs:
        bar = tqdm(
            pairs,
            total=len(pairs),
            desc="sync_cache",
            unit="room",
            dynamic_ncols=True,
        )
        iterator = bar
    results: list[SyncResult] = []
    try:
        for room, scene in iterator:
            if bar is not None:
                bar.set_postfix_str(room, refresh=False)
            result = sync_room(
                room,
                scene,
                cache,
                force=force,
                dry_run=dry_run,
                write_csv=write_csv,
            )
            results.append(result)
            if bar is not None and result.status == STATUS_FAILED:
                tqdm.write(f"  ! {result.room}: {result.error}")
    finally:
        if bar is not None:
            bar.close()
    return results


def _count_by_status(results: Iterable[SyncResult]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    return counts


def write_manifest(cache_dir: Path, results: Sequence[SyncResult]) -> Path:
    """Write a JSON audit log of the most recent sync into ``cache_dir``."""

    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_rooms": len(results),
        "counts": _count_by_status(results),
        "rooms": [asdict(r) for r in results],
    }
    path = cache_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


_STATUS_GLYPH = {
    STATUS_BUILT: "+",
    STATUS_SKIPPED: "=",
    STATUS_FAILED: "!",
    STATUS_WOULD_BUILD: "?+",
    STATUS_WOULD_SKIP: "?=",
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="graph.sync_cache",
        description=(
            "Discover SpatialLM scene.json exports and build/refresh the "
            "scene_graph.json cache used for GNN training."
        ),
    )
    parser.add_argument(
        "--scan-root",
        default=str(DEFAULT_SCAN_ROOT),
        help=f"Directory to scan for scene.json (default: {DEFAULT_SCAN_ROOT}).",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(DEFAULT_CACHE_DIR),
        help=f"Where to write scene_graph.json files (default: {DEFAULT_CACHE_DIR}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even when the cache is newer than the source scene.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without writing anything.",
    )
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Skip writing the edges.csv companion file beside each cached graph.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into nested directories under --scan-root instead of "
             "only inspecting immediate subdirectories.",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Skip writing manifest.json into --cache-dir.",
    )
    args = parser.parse_args(argv)

    scan_root = Path(args.scan_root).expanduser().resolve()
    cache_dir = Path(args.cache_dir).expanduser().resolve()

    if not scan_root.exists():
        print(f"Error: scan root not found: {scan_root}", file=sys.stderr)
        return 2

    results = sync_cache(
        scan_root,
        cache_dir,
        force=args.force,
        dry_run=args.dry_run,
        write_csv=not args.no_csv,
        recursive=args.recursive,
        show_progress=True,
    )

    if not results:
        hint = " (try --recursive)" if not args.recursive else ""
        print(f"No scene.json files found under {scan_root}{hint}.")
        return 0

    counts = _count_by_status(results)
    summary_bits = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"Sync complete: {len(results)} room(s) [{summary_bits}]")

    failures = [r for r in results if r.status == STATUS_FAILED]
    for r in failures:
        print(f"  ! {r.room:<32} -> {r.destination}\n      error: {r.error}")

    if args.dry_run:
        for r in results:
            marker = _STATUS_GLYPH.get(r.status, ".")
            print(f"  {marker} {r.room:<32} -> {r.destination}")

    if not args.dry_run and not args.no_manifest:
        manifest_path = write_manifest(cache_dir, results)
        print(f"Manifest: {manifest_path}")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "SyncResult",
    "DEFAULT_SCAN_ROOT",
    "DEFAULT_CACHE_DIR",
    "STATUS_BUILT",
    "STATUS_SKIPPED",
    "STATUS_FAILED",
    "STATUS_WOULD_BUILD",
    "STATUS_WOULD_SKIP",
    "discover_scenes",
    "is_stale",
    "sync_room",
    "sync_cache",
    "write_manifest",
    "main",
]
