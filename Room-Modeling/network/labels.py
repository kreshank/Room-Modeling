"""Principle vocabulary, status enum, and per-node-type applicability mask.

This module is the schema-of-record for what the GNN heads predict. It mirrors
the principle names emitted by `room_graph.evaluate_principles` so the GNN can
be a drop-in replacement (and later a learned approximation) of the rule
engine.
"""

from __future__ import annotations


PRINCIPLES: tuple[str, ...] = (
    "command_position",
    "solid_backing",
    "bed_aligned_with_door",
    "mirror_faces_bed",
    "clear_center",
    "sharp_corner_points_at_seat",
    "pairing_balance",
    "door_alignment",
    "light_window_proximity",
)

STATUSES: tuple[str, ...] = ("violated", "weak", "good")
STATUS_TO_INDEX: dict[str, int] = {s: i for i, s in enumerate(STATUSES)}
INDEX_TO_STATUS: dict[int, str] = {i: s for i, s in enumerate(STATUSES)}
STATUS_SCORE_AXIS: tuple[float, ...] = (0.0, 0.5, 1.0)


# Canonical label vocabulary. Drawn from `graph/config.py` plus the structural
# kinds emitted by the scene graph. Kept in a fixed order so embedding indices
# are stable across runs.
LABEL_VOCAB: tuple[str, ...] = (
    "desk",
    "dressing_table",
    "coffee_table",
    "dining_table",
    "side_table",
    "nightstand",
    "sofa",
    "couch",
    "chair",
    "armchair",
    "stool",
    "wardrobe",
    "tv_cabinet",
    "bookcase",
    "sideboard",
    "cupboard",
    "tv",
    "computer",
    "fireplace",
    "floor-standing_lamp",
    "table_lamp",
    "lamp",
    "chandelier",
    "mirror",
    "painting",
    "carpet",
    "curtain",
    "rug",
    "plants",
    "bed",
    "wall",
    "door",
    "window",
    "room",
    "seating_group",
    "bed_group",
    "desk_group",
)
LABEL_TO_INDEX: dict[str, int] = {l: i for i, l in enumerate(LABEL_VOCAB)}
UNKNOWN_LABEL_INDEX: int = len(LABEL_VOCAB)
LABEL_VOCAB_SIZE: int = len(LABEL_VOCAB) + 1


KIND_VOCAB: tuple[str, ...] = ("object", "wall", "door", "window", "room", "zone")
KIND_TO_INDEX: dict[str, int] = {k: i for i, k in enumerate(KIND_VOCAB)}


# Original edge type names emitted by `graph/scene_graph.py` and
# `graph/functional.py`. We collapse the heterogeneous edge buckets in
# `data.py` to the cross-product of `(src_kind, dst_kind)` and feed this
# fine-grained edge type back in as a one-hot in `edge_attr`, so the model
# still has access to the full information without paying for 28 separate
# convolutions.
EDGE_KIND_VOCAB: tuple[str, ...] = (
    "near",
    "faces",
    "overlaps",
    "parallel_to",
    "in_front_of",
    "behind",
    "left_of",
    "right_of",
    "table_for",
    "touching_wall",
    "against_wall",
    "parallel_to_wall",
    "has_backing",
    "door_in_wall",
    "window_in_wall",
    "inside_room",
    "outside_room",
    "focal_point_of",
    "participates_in",
    "blocks_path",
    "serves",
    "traffic_path",
    "entry_path",
)
EDGE_KIND_TO_INDEX: dict[str, int] = {k: i for i, k in enumerate(EDGE_KIND_VOCAB)}
UNKNOWN_EDGE_KIND_INDEX: int = len(EDGE_KIND_VOCAB)
EDGE_KIND_VOCAB_SIZE: int = len(EDGE_KIND_VOCAB) + 1


def edge_kind_to_index(edge_kind: str) -> int:
    return EDGE_KIND_TO_INDEX.get(edge_kind, UNKNOWN_EDGE_KIND_INDEX)


# (node_type, principle) applicability. A `True` cell means the rule engine can
# legitimately produce a check for nodes of that type / principle pair, and the
# GNN should output a prediction. Derived directly from which `target` ids each
# rule in `graph/fengshui.py` can emit.
NODE_TYPE_PRINCIPLE_MASK: dict[str, tuple[bool, ...]] = {
    "object": (
        True,   # command_position  (bed / desk / primary seat)
        True,   # solid_backing     (bed / primary seat)
        True,   # bed_aligned_with_door
        True,   # mirror_faces_bed
        False,  # clear_center      (room)
        True,   # sharp_corner_points_at_seat
        True,   # pairing_balance   (bed)
        False,  # door_alignment    (door)
        True,   # light_window_proximity
    ),
    "room": (
        False, False, False, False,
        True,   # clear_center
        False, False, False, False,
    ),
    "door": (
        False, False, False, False, False, False, False,
        True,   # door_alignment
        False,
    ),
    "window": (False,) * len(PRINCIPLES),
    "wall": (False,) * len(PRINCIPLES),
    "zone": (False,) * len(PRINCIPLES),
}

# Node types for which we instantiate prediction heads. Excludes types that
# have no applicable principles so we don't waste parameters.
HEAD_NODE_TYPES: tuple[str, ...] = ("object", "room", "door")


def applicable_mask(node_type: str) -> tuple[bool, ...]:
    return NODE_TYPE_PRINCIPLE_MASK.get(node_type, (False,) * len(PRINCIPLES))


def label_to_index(label: str | None) -> int:
    if not label:
        return UNKNOWN_LABEL_INDEX
    return LABEL_TO_INDEX.get(label.lower(), UNKNOWN_LABEL_INDEX)


__all__ = [
    "PRINCIPLES",
    "STATUSES",
    "STATUS_SCORE_AXIS",
    "STATUS_TO_INDEX",
    "INDEX_TO_STATUS",
    "LABEL_VOCAB",
    "LABEL_TO_INDEX",
    "UNKNOWN_LABEL_INDEX",
    "LABEL_VOCAB_SIZE",
    "KIND_VOCAB",
    "KIND_TO_INDEX",
    "EDGE_KIND_VOCAB",
    "EDGE_KIND_TO_INDEX",
    "EDGE_KIND_VOCAB_SIZE",
    "UNKNOWN_EDGE_KIND_INDEX",
    "NODE_TYPE_PRINCIPLE_MASK",
    "HEAD_NODE_TYPES",
    "applicable_mask",
    "label_to_index",
    "edge_kind_to_index",
]
