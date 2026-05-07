"""Parse SpatialLM layout text into an editable room-scene JSON schema.

SpatialLM layout text usually looks like:
    wall_0=Wall(ax,ay,az,bx,by,bz,height,thickness)
    door_0=Door(wall_0,position_x,position_y,position_z,width,height)
    window_0=Window(wall_0,position_x,position_y,position_z,width,height)
    bbox_0=Bbox(class_name,position_x,position_y,position_z,angle_z,scale_x,scale_y,scale_z)

The output JSON is meant to be consumed by viewer/index.html or by your own
floor-plan / feng-shui heuristics pipeline.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

EntityKind = Literal["wall", "door", "window", "furniture", "object", "unknown"]

_LINE_RE = re.compile(r"^\s*([a-zA-Z]+)_(\d+)\s*=\s*([A-Za-z]+)\((.*)\)\s*$")


def _split_params(params: str) -> list[str]:
    """Split a comma-separated parameter list.

    SpatialLM class labels are currently simple strings, so plain comma splitting
    is enough. The helper trims whitespace and ignores blank fragments.
    """

    return [p.strip() for p in params.split(",") if p.strip() != ""]


def _f(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _wall_id(value: str) -> str:
    value = value.strip()
    if value.startswith("wall_"):
        return value
    if value.isdigit():
        return f"wall_{value}"
    return value


def angle_from_points(ax: float, ay: float, bx: float, by: float) -> float:
    return math.atan2(by - ay, bx - ax)


def normalize_yaw_rad(yaw: float) -> float:
    return (yaw + math.pi) % (2 * math.pi) - math.pi


def yaw_to_deg(yaw_rad: float) -> float:
    return math.degrees(normalize_yaw_rad(yaw_rad))


@dataclass
class EditableEntity:
    id: str
    kind: EntityKind
    label: str
    x: float
    y: float
    z: float
    yaw_rad: float
    width: float
    depth: float
    height: float
    confirmed: bool = False
    confidence: float | None = None
    source: str = "spatiallm"
    raw: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["yaw_deg"] = yaw_to_deg(self.yaw_rad)
        return data


def parse_layout_text(layout_text: str) -> tuple[list[EditableEntity], list[dict[str, Any]]]:
    """Return normalized editable entities and any unparsable raw lines."""

    raw_walls: dict[str, dict[str, Any]] = {}
    raw_doors: list[dict[str, Any]] = []
    raw_windows: list[dict[str, Any]] = []
    raw_bboxes: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for line_number, line in enumerate(layout_text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _LINE_RE.match(stripped)
        if not match:
            skipped.append({"line_number": line_number, "line": line, "reason": "not a recognized SpatialLM entity"})
            continue

        prefix, num, ctor, params_s = match.groups()
        entity_id = f"{prefix.lower()}_{num}"
        ctor_l = ctor.lower()
        params = _split_params(params_s)

        try:
            if ctor_l == "wall" and len(params) >= 8:
                ax, ay, az, bx, by, bz, height, thickness = map(_f, params[:8])
                raw_walls[entity_id] = {
                    "id": entity_id,
                    "ax": ax,
                    "ay": ay,
                    "az": az,
                    "bx": bx,
                    "by": by,
                    "bz": bz,
                    "height": height,
                    "thickness": thickness,
                    "line_number": line_number,
                    "line": line,
                }
            elif ctor_l == "door" and len(params) >= 6:
                raw_doors.append({
                    "id": entity_id,
                    "wall_id": _wall_id(params[0]),
                    "position_x": _f(params[1]),
                    "position_y": _f(params[2]),
                    "position_z": _f(params[3]),
                    "width": _f(params[4]),
                    "height": _f(params[5]),
                    "line_number": line_number,
                    "line": line,
                })
            elif ctor_l == "window" and len(params) >= 6:
                raw_windows.append({
                    "id": entity_id,
                    "wall_id": _wall_id(params[0]),
                    "position_x": _f(params[1]),
                    "position_y": _f(params[2]),
                    "position_z": _f(params[3]),
                    "width": _f(params[4]),
                    "height": _f(params[5]),
                    "line_number": line_number,
                    "line": line,
                })
            elif ctor_l == "bbox" and len(params) >= 8:
                label = params[0]
                raw_bboxes.append({
                    "id": entity_id,
                    "class_name": label,
                    "position_x": _f(params[1]),
                    "position_y": _f(params[2]),
                    "position_z": _f(params[3]),
                    "angle_z": _f(params[4]),
                    "scale_x": abs(_f(params[5])),
                    "scale_y": abs(_f(params[6])),
                    "scale_z": abs(_f(params[7])),
                    "line_number": line_number,
                    "line": line,
                })
            else:
                skipped.append({"line_number": line_number, "line": line, "reason": f"unsupported constructor/arity: {ctor}"})
        except (TypeError, ValueError, KeyError, IndexError) as exc:
            skipped.append({"line_number": line_number, "line": line, "reason": repr(exc)})

    entities: list[EditableEntity] = []

    for wall in raw_walls.values():
        ax, ay, az, bx, by, bz = wall["ax"], wall["ay"], wall["az"], wall["bx"], wall["by"], wall["bz"]
        length = math.dist((ax, ay, az), (bx, by, bz))
        yaw = angle_from_points(ax, ay, bx, by)
        entities.append(
            EditableEntity(
                id=wall["id"],
                kind="wall",
                label="wall",
                x=(ax + bx) / 2.0,
                y=(ay + by) / 2.0,
                z=(az + bz) / 2.0 + wall["height"] / 2.0,
                yaw_rad=yaw,
                width=length,
                depth=max(wall["thickness"], 0.02),
                height=wall["height"],
                raw=wall,
            )
        )

    def add_fixture(kind: Literal["door", "window"], fixture: dict[str, Any]) -> None:
        wall = raw_walls.get(fixture["wall_id"])
        yaw = 0.0
        depth = 0.10
        if wall:
            yaw = angle_from_points(wall["ax"], wall["ay"], wall["bx"], wall["by"])
            depth = max(wall["thickness"], 0.08)
        entities.append(
            EditableEntity(
                id=fixture["id"],
                kind=kind,
                label=kind,
                x=fixture["position_x"],
                y=fixture["position_y"],
                z=fixture["position_z"],
                yaw_rad=yaw,
                width=fixture["width"],
                depth=depth,
                height=fixture["height"],
                raw=fixture,
            )
        )

    for door in raw_doors:
        add_fixture("door", door)
    for window in raw_windows:
        add_fixture("window", window)

    for bbox in raw_bboxes:
        label = bbox["class_name"]
        kind: EntityKind = "furniture" if label and label.lower() != "unknown" else "unknown"
        entities.append(
            EditableEntity(
                id=bbox["id"],
                kind=kind,
                label=label,
                x=bbox["position_x"],
                y=bbox["position_y"],
                z=bbox["position_z"],
                yaw_rad=normalize_yaw_rad(bbox["angle_z"]),
                width=bbox["scale_x"],
                depth=bbox["scale_y"],
                height=bbox["scale_z"],
                raw=bbox,
            )
        )

    return entities, skipped


def scene_bounds(entities: Iterable[EditableEntity], padding: float = 0.5) -> dict[str, float]:
    xs: list[float] = []
    ys: list[float] = []
    for e in entities:
        c, s = abs(math.cos(e.yaw_rad)), abs(math.sin(e.yaw_rad))
        half_x = 0.5 * (e.width * c + e.depth * s)
        half_y = 0.5 * (e.width * s + e.depth * c)
        xs.extend([e.x - half_x, e.x + half_x])
        ys.extend([e.y - half_y, e.y + half_y])
    if not xs or not ys:
        return {"min_x": -1, "max_x": 1, "min_y": -1, "max_y": 1}
    return {
        "min_x": min(xs) - padding,
        "max_x": max(xs) + padding,
        "min_y": min(ys) - padding,
        "max_y": max(ys) + padding,
    }


def build_scene(
    layout_text: str,
    *,
    source_ply: str | None = None,
    source_layout_txt: str | None = None,
    model_path: str | None = None,
) -> dict[str, Any]:
    entities, skipped = parse_layout_text(layout_text)
    return {
        "schema_version": "room_scene_v1",
        "units": "meters",
        "coordinate_system": {
            "up_axis": "z",
            "topdown_axes": ["x", "y"],
            "yaw_axis": "z",
            "yaw_units": "radians",
        },
        "source": {
            "ply_file": source_ply,
            "spatiallm_layout_txt": source_layout_txt,
            "model_path": model_path,
        },
        "bounds": scene_bounds(entities),
        "entities": [e.to_dict() for e in entities],
        "skipped_lines": skipped,
    }


def parse_layout_file(
    layout_txt_path: str | Path,
    json_out_path: str | Path | None = None,
    *,
    source_ply: str | None = None,
    model_path: str | None = None,
) -> dict[str, Any]:
    layout_txt_path = Path(layout_txt_path)
    layout_text = layout_txt_path.read_text(encoding="utf-8")
    scene = build_scene(
        layout_text,
        source_ply=source_ply,
        source_layout_txt=str(layout_txt_path),
        model_path=model_path,
    )
    if json_out_path:
        json_out_path = Path(json_out_path)
        json_out_path.parent.mkdir(parents=True, exist_ok=True)
        json_out_path.write_text(json.dumps(scene, indent=2), encoding="utf-8")
    return scene
