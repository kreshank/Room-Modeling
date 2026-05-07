"""Canonical vocab for principles/actions/targets and topic filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


CANONICAL_PRINCIPLES: dict[str, list[str]] = {
    "command_position": ["command position", "see the door", "view of the door", "visible from entry"],
    "solid_backing": ["against the wall", "backed by wall", "solid wall behind", "wall backing"],
    "bed_aligned_with_door": ["bed facing door", "bed aligned with door", "bed in line with door"],
    "mirror_faces_bed": ["mirror facing bed", "mirror reflects bed"],
    "clear_center": ["clear the center", "open center", "center clutter"],
    "sharp_corner_points_at_seat": ["sharp corner", "poison arrow", "corner points at"],
    "pairing_balance": ["nightstands on both sides", "balanced pair", "symmetry around bed"],
    "door_alignment": ["door alignment", "doors in line", "straight line from door"],
    "light_window_proximity": ["near window", "natural light", "window proximity"],
    "bed_under_window": ["bed under window", "headboard under window"],
    "bed_under_beam": ["bed under beam"],
    "clutter_under_bed": ["clutter under bed", "storage under bed"],
    "shoes_at_door": ["shoes at door", "entry clutter"],
    "water_feature_placement": ["water feature", "aquarium placement", "fountain placement"],
    "plants_in_corner": ["plants in corner", "corner plant"],
    "color_scheme": ["color palette", "colors in room", "paint color"],
    "broken_items": ["broken items", "fix broken", "remove broken"],
}

ACTION_VERBS = {
    "move", "rotate", "remove", "declutter", "add", "place", "swap", "reorient",
    "clear", "push", "pull", "center", "align", "turn",
}

TARGET_OBJECTS = {
    "bed", "desk", "chair", "sofa", "couch", "coffee_table", "dining_table", "side_table",
    "nightstand", "wardrobe", "tv_cabinet", "bookcase", "sideboard", "cupboard", "mirror",
    "door", "window", "tv", "computer", "floor-standing_lamp", "chandelier", "painting",
    "plants", "carpet", "curtain", "rug", "entry",
}

TOPIC_KEYWORDS = {
    "feng shui", "layout", "room", "bedroom", "living room", "office",
    "couch", "sofa", "bed", "desk", "mirror", "door", "window", "tv",
    "entry", "clutter", "furniture",
}


@dataclass
class TopicResult:
    is_on_topic: bool
    matched_keywords: list[str]


def topic_filter(title: str, transcript: str) -> TopicResult:
    text = f"{title} {transcript}".lower()
    matched = [k for k in TOPIC_KEYWORDS if k in text]
    return TopicResult(is_on_topic=bool(matched), matched_keywords=sorted(set(matched)))

