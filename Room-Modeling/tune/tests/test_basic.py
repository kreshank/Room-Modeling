from __future__ import annotations

from pathlib import Path

from tune.clean import clean_record
from tune.ingest import ingest_xlsx_dir
from tune.rule_extract import extract_matches_for_doc
from tune.vocab import topic_filter


DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "cliff_transcripts"


def test_ingest_2022_count_at_least_980():
    rows, stats = ingest_xlsx_dir(DATA_DIR)
    rows_2022 = [r for r in rows if r.get("source_xlsx") == "dm_shorts_2022.xlsx"]
    assert len(rows_2022) >= 980


def test_topic_filter_dong_zhi_off_topic():
    title = "Dong zhi falls on 22 Dec for 2022 it's celebrated by many countries in East Asia not just China"
    transcript = "oh happy don't you which celebrates the shortest day of the year unlike most holidays"
    topic = topic_filter(title, transcript)
    assert not topic.is_on_topic


def test_rule_extractor_on_synthetic_transcript():
    rec = {
        "doc_id": "x",
        "title": "living room fix",
        "transcript_raw": "place the couch against the wall so you can see the door",
    }
    clean = clean_record(rec)
    out = extract_matches_for_doc(clean)
    principles = {m["principle"] for m in out["principles"]}
    assert "solid_backing" in principles
    assert "command_position" in principles


