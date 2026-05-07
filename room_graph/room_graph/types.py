"""Lightweight typed records used across room_graph stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

NodeType = Literal["wall", "door", "window", "object", "room", "zone", "unknown"]
PrincipleStatus = Literal["good", "weak", "violated", "not_applicable"]


@dataclass
class OrientedRect:
    """Top-down (XY-plane) oriented rectangle for an entity."""

    cx: float
    cy: float
    width: float
    depth: float
    yaw_rad: float
    z: float = 0.0
    height: float = 0.0

    @property
    def center(self) -> tuple[float, float]:
        return (self.cx, self.cy)

    @property
    def half_size(self) -> tuple[float, float]:
        return (0.5 * self.width, 0.5 * self.depth)


@dataclass
class Node:
    id: str
    type: NodeType
    label: str
    geometry: dict[str, Any] = field(default_factory=dict)
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    source: str
    target: str
    type: str
    features: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"source": self.source, "target": self.target, "type": self.type}
        out.update(self.features)
        return out


@dataclass
class Zone:
    id: str
    type: str
    members: list[str]
    centroid_xy: tuple[float, float] | None = None
    attrs: dict[str, Any] = field(default_factory=dict)


@dataclass
class PrincipleCheck:
    principle: str
    target: str
    status: PrincipleStatus
    score: float
    evidence: list[str] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "principle": self.principle,
            "target": self.target,
            "status": self.status,
            "score": self.score,
            "evidence": list(self.evidence),
            "edges": list(self.edges),
        }
