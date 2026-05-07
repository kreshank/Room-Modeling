# room_tune

Transcript ingestion and weak-supervision extraction pipeline for `dm_shorts_*.xlsx`.

## Pipeline

```text
xlsx -> raw.jsonl -> clean.jsonl -> rule_extracted.jsonl -> final.jsonl -> dataset.jsonl + summary.json
```

## Quick start

```bash
python -m tune --in ../data/cliff_transcripts --out ../outs/transcripts
```
