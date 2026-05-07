"""Tunable thresholds. Centralized so rules stay deterministic and inspectable."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class GraphConfig:
    grid_snap_m: float = 0.02
    near_distance_m: float = 1.0
    far_distance_m: float = 3.0
    touching_wall_m: float = 0.05
    against_wall_m: float = 0.30
    parallel_yaw_tol_deg: float = 15.0
    facing_angle_tol_deg: float = 25.0
    overlap_eps_m2: float = 1e-3

    has_backing_min_coverage: float = 0.80
    has_backing_max_clearance_m: float = 0.30

    los_opaque_min_height_m: float = 1.40

    walkable_grid_cell_m: float = 0.05
    corridor_min_width_m: float = 0.60

    seating_cluster_radius_m: float = 1.5
    bed_nightstand_max_dist_m: float = 0.6
    desk_chair_max_dist_m: float = 0.8
    table_for_max_dist_m: float = 0.6

    command_min_door_dist_m: float = 1.0
    command_door_alignment_deg: float = 20.0
    door_alignment_deg: float = 20.0
    sharp_corner_max_dist_m: float = 1.0
    sharp_corner_angle_deg: float = 30.0
    clear_center_radius_m: float = 1.0
    clear_center_max_occupancy: float = 0.25
    light_window_max_dist_m: float = 3.0


SOFT_LABELS: frozenset[str] = frozenset({
    "carpet",
    "curtain",
    "painting",
    "mirror",
    "chandelier",
    "floor-standing_lamp",
    "plants",
    "computer",
})

LOW_HEIGHT_LABELS: frozenset[str] = frozenset({
    "carpet",
    "stool",
    "side_table",
    "coffee_table",
    "nightstand",
})

SEAT_LABELS: frozenset[str] = frozenset({
    "sofa",
    "couch",
    "chair",
    "armchair",
    "stool",
})

PRIMARY_SEAT_LABELS: frozenset[str] = frozenset({
    "sofa",
    "couch",
    "armchair",
})

TABLE_LABELS: frozenset[str] = frozenset({
    "coffee_table",
    "dining_table",
    "side_table",
})

BED_LABELS: frozenset[str] = frozenset({"bed"})

DESK_LABELS: frozenset[str] = frozenset({"desk", "dressing_table"})

NIGHTSTAND_LABELS: frozenset[str] = frozenset({"nightstand"})

LAMP_LABELS: frozenset[str] = frozenset({
    "floor-standing_lamp",
    "table_lamp",
    "lamp",
    "chandelier",
})

FOCAL_LABEL_PRIORITY: tuple[str, ...] = (
    "tv",
    "fireplace",
)

_DEFAULT_FORWARD_AXIS: dict[str, tuple[float, float]] = {
    "sofa": (0.0, 1.0),
    "couch": (0.0, 1.0),
    "chair": (0.0, 1.0),
    "armchair": (0.0, 1.0),
    "stool": (0.0, 1.0),
    "bed": (0.0, 1.0),
    "desk": (0.0, 1.0),
    "dressing_table": (0.0, 1.0),
    "tv": (0.0, 1.0),
    "tv_cabinet": (0.0, 1.0),
    "computer": (0.0, 1.0),
    "mirror": (0.0, 1.0),
    "painting": (0.0, 1.0),
    "fireplace": (0.0, 1.0),
}


def forward_axis_for_label(label: str) -> tuple[float, float]:
    """Local-frame forward direction (unit vector) for a given object label.

    Returned as `(lx, ly)` in the object's local coordinate system before yaw.
    Default convention: front face is along +y. Tweak per label as needed.
    """

    return _DEFAULT_FORWARD_AXIS.get(label.lower(), (0.0, 1.0))

