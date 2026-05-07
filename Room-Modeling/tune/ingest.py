"""Read dm_shorts_*.xlsx files with stdlib-only XML parsing."""

from __future__ import annotations

import hashlib
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from xml.etree import ElementTree as ET

NS = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

HEADER_MAP = {
    "video date": "date",
    "video title": "title",
    "video transcript": "transcript_raw",
    "video url": "url",
    "transcript status": "status",
}


@dataclass
class IngestStats:
    files: int = 0
    rows_seen: int = 0
    rows_emitted: int = 0


def _cell_text(cell: ET.Element, strings: list[str]) -> str:
    ctype = cell.get("t")
    value = cell.find("main:v", NS)
    if ctype == "s":
        if value is None or value.text is None:
            return ""
        try:
            return strings[int(value.text)]
        except Exception:
            return ""
    if ctype == "inlineStr":
        is_el = cell.find("main:is", NS)
        if is_el is None:
            return ""
        return "".join(t.text or "" for t in is_el.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"))
    if value is None or value.text is None:
        return ""
    return value.text


def _col_letter(ref: str | None) -> str:
    if not ref:
        return ""
    m = re.match(r"([A-Z]+)\d+", ref)
    return m.group(1) if m else ""


def _read_sheet(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with zipfile.ZipFile(path) as zf:
        shared_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
        sheet_root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))

    strings: list[str] = []
    for si in shared_root.findall("main:si", NS):
        text = "".join(t.text or "" for t in si.iter("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"))
        strings.append(text)

    rows = sheet_root.find("main:sheetData", NS)
    if rows is None:
        return [], []
    row_nodes = rows.findall("main:row", NS)
    if not row_nodes:
        return [], []

    header_cells = row_nodes[0].findall("main:c", NS)
    headers: dict[str, str] = {}
    ordered_headers: list[str] = []
    for cell in header_cells:
        col = _col_letter(cell.get("r"))
        text = _cell_text(cell, strings).strip().lower()
        headers[col] = text
        ordered_headers.append(text)

    body: list[dict[str, str]] = []
    for row in row_nodes[1:]:
        item: dict[str, str] = {}
        for cell in row.findall("main:c", NS):
            col = _col_letter(cell.get("r"))
            header = headers.get(col, "")
            if not header:
                continue
            item[header] = _cell_text(cell, strings).strip()
        body.append(item)
    return ordered_headers, body


def stable_doc_id(url: str, date: str, title: str, row_index: int) -> str:
    payload = f"{url}|{date}|{title}|{row_index}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def ingest_xlsx_dir(input_dir: str | Path) -> tuple[list[dict[str, Any]], IngestStats]:
    base = Path(input_dir)
    files = sorted(base.glob("dm_shorts_*.xlsx"))
    stats = IngestStats(files=len(files))
    out: list[dict[str, Any]] = []

    for path in files:
        year_match = re.search(r"(\d{4})", path.name)
        year = int(year_match.group(1)) if year_match else None
        _headers, rows = _read_sheet(path)
        for idx, raw in enumerate(rows, start=2):
            stats.rows_seen += 1
            mapped = {HEADER_MAP.get(k, k): v for k, v in raw.items()}
            title = mapped.get("title", "")
            transcript = mapped.get("transcript_raw", "")
            rec = {
                "doc_id": stable_doc_id(mapped.get("url", ""), mapped.get("date", ""), title, idx),
                "year": year,
                "date": mapped.get("date", ""),
                "title": title,
                "transcript_raw": transcript,
                "url": mapped.get("url", ""),
                "status": mapped.get("status", ""),
                "source_xlsx": path.name,
                "row_index": idx,
            }
            out.append(rec)
            stats.rows_emitted += 1

    return out, stats
