"""Tests for the scene_graph cache sync workflow."""

from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from graph.sync_cache import (  # noqa: E402  (sys.path tweak above is intentional)
    STATUS_BUILT,
    STATUS_SKIPPED,
    STATUS_WOULD_BUILD,
    STATUS_WOULD_SKIP,
    discover_scenes,
    main,
    sync_cache,
)


SAMPLE_SCENE = (
    PROJECT_ROOT / "outs" / "spatial_editor_outputs" / "my_room" / "scene.json"
)


@pytest.fixture(scope="module")
def sample_scene_path() -> Path:
    if not SAMPLE_SCENE.exists():
        pytest.skip(f"sample scene fixture missing: {SAMPLE_SCENE}")
    return SAMPLE_SCENE


def _make_room(scan_root: Path, name: str, scene_src: Path) -> Path:
    room = scan_root / name
    room.mkdir(parents=True, exist_ok=True)
    dest = room / "scene.json"
    shutil.copy(scene_src, dest)
    # Anchor the source mtime well in the past so the first sync sees the
    # cache as newer once it is written, regardless of filesystem precision.
    past = time.time() - 60
    os.utime(dest, (past, past))
    return dest


def test_discover_scenes_default_one_level(tmp_path, sample_scene_path):
    scan = tmp_path / "scan"
    _make_room(scan, "room_a", sample_scene_path)
    _make_room(scan, "room_b", sample_scene_path)
    # Nested scene.json that should be IGNORED in default mode.
    nested = scan / "nested" / "deep" / "scene.json"
    nested.parent.mkdir(parents=True)
    shutil.copy(sample_scene_path, nested)

    pairs = discover_scenes(scan, recursive=False)
    rooms = [room for room, _ in pairs]
    assert rooms == ["room_a", "room_b"]


def test_discover_scenes_recursive_includes_nested(tmp_path, sample_scene_path):
    scan = tmp_path / "scan"
    _make_room(scan, "room_a", sample_scene_path)
    nested = scan / "nested" / "deep" / "scene.json"
    nested.parent.mkdir(parents=True)
    shutil.copy(sample_scene_path, nested)

    pairs = discover_scenes(scan, recursive=True)
    rooms = sorted(room for room, _ in pairs)
    assert rooms == ["nested/deep", "room_a"]


def test_sync_builds_then_skips_then_rebuilds_on_touch(tmp_path, sample_scene_path):
    scan = tmp_path / "scan"
    cache = tmp_path / "cache"
    scene_path = _make_room(scan, "room_a", sample_scene_path)

    first = sync_cache(scan, cache)
    assert [r.status for r in first] == [STATUS_BUILT]
    cached_graph = cache / "room_a" / "scene_graph.json"
    cached_csv = cache / "room_a" / "edges.csv"
    assert cached_graph.exists()
    assert cached_csv.exists()
    payload = json.loads(cached_graph.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "scene_graph_v1"

    second = sync_cache(scan, cache)
    assert [r.status for r in second] == [STATUS_SKIPPED]

    bumped = time.time() + 5
    os.utime(scene_path, (bumped, bumped))

    third = sync_cache(scan, cache)
    assert [r.status for r in third] == [STATUS_BUILT]


def test_sync_force_rebuilds_even_when_fresh(tmp_path, sample_scene_path):
    scan = tmp_path / "scan"
    cache = tmp_path / "cache"
    _make_room(scan, "room_a", sample_scene_path)

    sync_cache(scan, cache)
    forced = sync_cache(scan, cache, force=True)
    assert [r.status for r in forced] == [STATUS_BUILT]


def test_sync_dry_run_writes_nothing(tmp_path, sample_scene_path):
    scan = tmp_path / "scan"
    cache = tmp_path / "cache"
    _make_room(scan, "room_a", sample_scene_path)

    results = sync_cache(scan, cache, dry_run=True)
    assert [r.status for r in results] == [STATUS_WOULD_BUILD]
    assert not cache.exists() or not any(cache.rglob("scene_graph.json"))

    sync_cache(scan, cache)
    dry_again = sync_cache(scan, cache, dry_run=True)
    assert [r.status for r in dry_again] == [STATUS_WOULD_SKIP]


def test_sync_no_csv_skips_companion(tmp_path, sample_scene_path):
    scan = tmp_path / "scan"
    cache = tmp_path / "cache"
    _make_room(scan, "room_a", sample_scene_path)

    sync_cache(scan, cache, write_csv=False)
    assert (cache / "room_a" / "scene_graph.json").exists()
    assert not (cache / "room_a" / "edges.csv").exists()


def test_main_writes_manifest_and_returns_zero(tmp_path, sample_scene_path, capsys):
    scan = tmp_path / "scan"
    cache = tmp_path / "cache"
    _make_room(scan, "room_a", sample_scene_path)
    _make_room(scan, "room_b", sample_scene_path)

    rc = main(["--scan-root", str(scan), "--cache-dir", str(cache)])
    assert rc == 0
    capsys.readouterr()  # drain stdout

    manifest_path = cache / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["n_rooms"] == 2
    assert manifest["counts"].get(STATUS_BUILT) == 2
    rooms = sorted(r["room"] for r in manifest["rooms"])
    assert rooms == ["room_a", "room_b"]


def test_main_handles_empty_scan_root(tmp_path, capsys):
    scan = tmp_path / "empty"
    scan.mkdir()
    cache = tmp_path / "cache"
    rc = main(["--scan-root", str(scan), "--cache-dir", str(cache)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "No scene.json files found" in out
