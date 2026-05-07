"""Deterministic transcript -> principle/action/target match extraction."""

from __future__ import annotations

import re
from typing import Any

from .vocab import ACTION_VERBS, CANONICAL_PRINCIPLES, TARGET_OBJECTS, topic_filter


def _find_all_spans(text: str, needle: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    while True:
        i = text.find(needle, start)
        if i < 0:
            break
        spans.append((i, i + len(needle)))
        start = i + len(needle)
    return spans


def _extract_action(window_text: str) -> str | None:
    words = re.findall(r"[a-zA-Z_'-]+", window_text.lower())
    for w in words:
        if w in ACTION_VERBS:
            return w
    return None


def _extract_target(window_text: str) -> str | None:
    words = re.findall(r"[a-zA-Z_'-]+", window_text.lower())
    for w in words:
        if w in TARGET_OBJECTS:
            return w
    joined = " ".join(words)
    if "living room" in joined:
        return "living_room"
    return None


def _polarity(window_text: str) -> str:
    low = window_text.lower()
    if any(x in low for x in ["don't", "dont", "never", "avoid", "not "]):
        return "avoid"
    return "promote"


def extract_matches_for_doc(doc: dict[str, Any]) -> dict[str, Any]:
    title = str(doc.get("title", ""))
    transcript = str(doc.get("transcript_clean", doc.get("transcript_raw", "")))
    full_text = f"{title}. {transcript}".strip().lower()
    matches: list[dict[str, Any]] = []

    for principle, phrases in CANONICAL_PRINCIPLES.items():
        for phrase in phrases:
            needle = phrase.lower()
            for s, e in _find_all_spans(full_text, needle):
                left = max(0, s - 80)
                right = min(len(full_text), e + 80)
                window = full_text[left:right]
                action = _extract_action(window)
                target = _extract_target(window)
                confidence = 0.75
                if action:
                    confidence += 0.1
                if target:
                    confidence += 0.1
                confidence = min(0.98, confidence)
                matches.append(
                    {
                        "principle": principle,
                        "polarity": _polarity(window),
                        "target": target,
                        "action": action,
                        "evidence_span": full_text[s:e],
                        "char_start": s,
                        "char_end": e,
                        "confidence": round(confidence, 3),
                    }
                )

    topic = topic_filter(title, transcript)
    max_conf = max((m["confidence"] for m in matches), default=0.0)
    needs_llm = (topic.is_on_topic and not matches) or (topic.is_on_topic and max_conf < 0.3)

    out = dict(doc)
    out["topic"] = {
        "is_on_topic": topic.is_on_topic,
        "matched_keywords": topic.matched_keywords,
    }
    out["principles"] = matches
    out["rule_confidence"] = round(max_conf, 3)
    out["needs_llm"] = bool(needs_llm)
    return out


def extract_matches(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [extract_matches_for_doc(r) for r in rows]

