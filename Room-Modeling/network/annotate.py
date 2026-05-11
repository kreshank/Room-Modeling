"""LLM-based feng shui annotator.

Reads a predict JSON produced by ``network.cli predict``, compresses the
GNN output into a structured prompt, calls a local Ollama model, and emits
an extras JSON that the visualizer can load via the extras file picker.

Usage::

    python -m network.cli annotate \\
        --predict outs/inference/predict_my_room_<ts>.json

Ollama must be running (``ollama serve``) with the model pulled::

    ollama pull qwen2.5:0.5b
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

_SYSTEM_PROMPT = """\
You are a feng shui spatial analyst. Evaluate a room using these nine principles:

1. command_position: The main furniture (bed/sofa/desk) must face the entry door, not sit directly in the door's axis, and have a solid wall behind it.
2. solid_backing: Beds and primary seats need a solid wall behind them — not a door or window opening.
3. bed_aligned_with_door: The bed must not lie on the direct straight-line axis of any door.
4. mirror_faces_bed: Mirrors must not directly face or reflect the sleeping area.
5. clear_center: The central floor area of the room must stay free of large furniture.
6. sharp_corner_points_at_seat: No sharp furniture corner should point toward the primary seat within arm's reach.
7. pairing_balance: A bed should have a nightstand on both sides for energetic balance.
8. door_alignment: Two doors on opposing or parallel walls must not be directly aligned (creates a draining qi corridor).
9. light_window_proximity: The primary sleeping or seating area benefits from being near a natural light source.

Severity: "violated" = clear breach, "weak" = partial or marginal issue, "good" = satisfied.\
"""

_IMPACT_MAP = {
    "command_position": "high",
    "solid_backing": "high",
    "bed_aligned_with_door": "high",
    "mirror_faces_bed": "medium",
    "clear_center": "medium",
    "sharp_corner_points_at_seat": "medium",
    "pairing_balance": "low",
    "door_alignment": "medium",
    "light_window_proximity": "low",
}

_SCORE_LABEL = {
    (0.85, 1.01): "Excellent",
    (0.65, 0.85): "Good",
    (0.45, 0.65): "Fair",
    (0.25, 0.45): "Poor",
    (0.0, 0.25): "Very Poor",
}


def _score_label(graph_score: float) -> str:
    for (lo, hi), label in _SCORE_LABEL.items():
        if lo <= graph_score < hi:
            return label
    return "Unknown"


def _aggregate_by_principle(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per principle: pick worst status, then lowest score target."""
    from collections import defaultdict
    by_principle: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in predictions:
        if p.get("status") in ("violated", "weak"):
            by_principle[p["principle"]].append(p)
    rows = []
    status_rank = {"violated": 0, "weak": 1}
    for principle, items in by_principle.items():
        worst = min(items, key=lambda x: (status_rank.get(x["status"], 2), x.get("score", 1.0)))
        count = len(items)
        rows.append({
            "principle": principle,
            "target": worst["target"],
            "status": worst["status"],
            "score": worst.get("score", 0.0),
            "affected_count": count,
        })
    rows.sort(key=lambda x: (status_rank.get(x["status"], 2), x["score"]))
    return rows


def _build_user_message(predict: dict[str, Any]) -> str:
    graph_score = predict.get("graph_score", 0.0)
    predictions = predict.get("principle_predictions") or []

    aggregated = _aggregate_by_principle(predictions)
    if not aggregated:
        table = "All principles scored as good."
    else:
        rows = [
            f"| {p['principle']} | {p['target']} | {p['status']} | {p['score']:.2f} | {p['affected_count']} node(s) |"
            for p in aggregated
        ]
        table = (
            "| principle | worst_target | status | score | affected |\n"
            "|-----------|--------------|--------|-------|----------|\n"
            + "\n".join(rows)
        )

    schema_template = json.dumps({
        "summary": "<2-3 sentence plain-English room summary>",
        "recommendations": ["<action 1>", "<action 2>", "<action 3>"],
        "ranked_violations": [
            {"principle": "<name>", "target": "<id>", "score": 0.0,
             "status": "<status>", "impact": "<high|medium|low>"}
        ],
        "explanations": [
            {"target": "<id>", "principle": "<name>",
             "status": "<status>", "score": 0.0, "text": "<1-2 sentence explanation>"}
        ],
    }, indent=2)

    return (
        f"Room graph_score: {graph_score:.2f} ({_score_label(graph_score)})\n\n"
        f"Non-good principle results:\n{table}\n\n"
        f"Respond with ONLY valid JSON matching this schema exactly:\n{schema_template}"
    )


def _validate_extras(raw: dict[str, Any], predict: dict[str, Any]) -> dict[str, Any]:
    """Fill in safe defaults for any keys the LLM omitted."""
    predictions = predict.get("principle_predictions") or []
    graph_score = float(predict.get("graph_score") or 0.0)
    non_good = [p for p in predictions if p.get("status") in ("violated", "weak")]

    summary = raw.get("summary")
    if not isinstance(summary, str) or not summary.strip():
        count_v = sum(1 for p in non_good if p.get("status") == "violated")
        count_w = sum(1 for p in non_good if p.get("status") == "weak")
        summary = (
            f"Room score: {graph_score:.2f} ({_score_label(graph_score)}). "
            f"{count_v} violation(s) and {count_w} warning(s) detected."
        )

    recommendations = raw.get("recommendations")
    if not isinstance(recommendations, list):
        recommendations = [
            f"Address the '{p['principle']}' issue on {p['target']}."
            for p in non_good[:3]
        ]

    ranked_violations = raw.get("ranked_violations")
    if not isinstance(ranked_violations, list):
        ranked_violations = [
            {
                "principle": p["principle"],
                "target": p["target"],
                "score": p.get("score", 0.0),
                "status": p["status"],
                "impact": _IMPACT_MAP.get(p["principle"], "medium"),
            }
            for p in sorted(non_good, key=lambda x: x.get("score", 0))
        ]
    else:
        for row in ranked_violations:
            if "impact" not in row:
                row["impact"] = _IMPACT_MAP.get(row.get("principle", ""), "medium")

    explanations = raw.get("explanations")
    if not isinstance(explanations, list):
        explanations = [
            {
                "target": p["target"],
                "principle": p["principle"],
                "status": p["status"],
                "score": p.get("score", 0.0),
                "text": f"{p['principle'].replace('_', ' ').title()} is {p['status']} for {p['target']}.",
            }
            for p in non_good
        ]

    return {
        "summary": summary,
        "recommendations": recommendations,
        "ranked_violations": ranked_violations,
        "explanations": explanations,
    }


def annotate(
    predict: dict[str, Any],
    *,
    model: str = "qwen2.5:0.5b",
    source_predict_path: str | None = None,
) -> dict[str, Any]:
    """Run the LLM annotator on a predict dict and return the extras dict.

    Raises ``ImportError`` if ``ollama`` is not installed.
    Raises ``RuntimeError`` if Ollama is unreachable or the model is not pulled.
    """
    try:
        import ollama  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "The 'ollama' Python package is required. Install it with: pip install ollama"
        ) from exc

    user_msg = _build_user_message(predict)

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            format="json",
        )
    except Exception as exc:
        raise RuntimeError(
            f"Ollama call failed. Is 'ollama serve' running and '{model}' pulled? "
            f"Error: {exc}"
        ) from exc

    raw_text = response.get("message", {}).get("content", "{}")
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw_text, re.DOTALL)
        try:
            raw = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            raw = {}

    validated = _validate_extras(raw, predict)
    graph_score = float(predict.get("graph_score") or 0.0)
    return {
        "schema_version": "fengshui_extras_v1",
        "source_predict": source_predict_path or "",
        "model": model,
        "graph_score": round(graph_score, 4),
        "overall_score_label": _score_label(graph_score),
        **validated,
    }


def add_annotate_arguments(parser: Any) -> None:
    parser.add_argument(
        "--predict",
        required=True,
        help="Path to a predict_*.json from 'network.cli predict'.",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5:0.5b",
        help="Ollama model tag. Default: qwen2.5:0.5b",
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Path to write extras JSON. Defaults to "
            "outs/inference/extras_<stem>_<timestamp>.json."
        ),
    )


def _resolve_extras_out_path(provided: str | None, predict_path: Path) -> Path:
    if provided:
        return Path(provided).expanduser()
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_stem = re.sub(r"[^A-Za-z0-9_-]+", "_", predict_path.stem or "predict")
    # Strip leading "predict_" prefix if present so we get "extras_my_room_…"
    safe_stem = re.sub(r"^predict_?", "", safe_stem)
    return predict_path.parent / f"extras_{safe_stem}_{timestamp}.json"


def run_annotate(args: Any) -> int:
    predict_path = Path(args.predict).expanduser()
    if not predict_path.exists():
        print(f"Error: predict JSON not found: {predict_path}")
        return 2

    predict_data = json.loads(predict_path.read_text(encoding="utf-8"))

    print(f"Annotating {predict_path.name} with model '{args.model}' …")

    try:
        extras = annotate(
            predict_data,
            model=args.model,
            source_predict_path=str(predict_path),
        )
    except (ImportError, RuntimeError) as exc:
        print(f"Error: {exc}")
        return 1

    out_path = _resolve_extras_out_path(args.out, predict_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(extras, indent=2), encoding="utf-8")

    print(f"Overall score: {extras['graph_score']} — {extras['overall_score_label']}")
    print(f"Summary: {extras['summary']}")
    print(f"Recommendations ({len(extras['recommendations'])}):")
    for i, rec in enumerate(extras["recommendations"], 1):
        print(f"  {i}. {rec}")
    print(f"Extras JSON: {out_path}")
    return 0


__all__ = [
    "annotate",
    "add_annotate_arguments",
    "run_annotate",
]
