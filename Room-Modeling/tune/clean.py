"""Text normalization, sentence handling, and simple quality flags."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

MUSIC_RE = re.compile(r"\[(music|applause|laughter|noise)\]", re.I)
WS_RE = re.compile(r"\s+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

COMMON_ENGLISH = {
    "the", "and", "to", "of", "a", "in", "is", "it", "you", "that", "for",
    "on", "with", "this", "be", "as", "are", "was", "at", "have", "your",
}


def normalize_text(text: str) -> str:
    text = text or ""
    text = MUSIC_RE.sub(" ", text)
    text = text.replace("\u2019", "'")
    text = WS_RE.sub(" ", text).strip()
    return text


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    parts = [p.strip() for p in SENTENCE_SPLIT_RE.split(text) if p.strip()]
    if not parts:
        return [text.strip()]
    return parts


def dedup_consecutive_sentences(sentences: list[str]) -> list[str]:
    out: list[str] = []
    prev = None
    for s in sentences:
        if prev is not None and s.lower() == prev.lower():
            continue
        out.append(s)
        prev = s
    return out


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def english_ratio(text: str) -> float:
    words = _word_tokens(text)
    if not words:
        return 0.0
    cnt = Counter(words)
    common_hits = sum(cnt[w] for w in COMMON_ENGLISH if w in cnt)
    return common_hits / max(1, len(words))


def quality_flags(text: str) -> dict[str, bool]:
    words = _word_tokens(text)
    n = len(words)
    ratio = english_ratio(text)
    return {
        "too_short": n < 20,
        "too_long": n > 800,
        "non_english": ratio < 0.02 and n > 30,
    }


def clean_record(rec: dict[str, Any]) -> dict[str, Any]:
    title = normalize_text(str(rec.get("title", "")))
    transcript = normalize_text(str(rec.get("transcript_raw", "")))
    sentences = dedup_consecutive_sentences(split_sentences(transcript))
    clean_t = " ".join(sentences)

    out = dict(rec)
    out["title"] = title
    out["transcript_raw"] = transcript
    out["transcript_clean"] = clean_t
    out["sentences"] = sentences
    out["quality_flags"] = quality_flags(clean_t)
    return out


def clean_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for rec in rows:
        c = clean_record(rec)
        key = (c.get("title", "").lower(), c.get("transcript_clean", "").lower())
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out
