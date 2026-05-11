# tune/ — Transcript ingestion + weak-supervision extraction

Reads `dm_shorts_*.xlsx` transcripts of fengshui short-form videos, cleans
them, runs principle / action / target rule extraction, and aggregates the
results into per-principle counts. The counts are consumed by
[`network.cli train`](Room-Modeling/network/README.md) as `log(1+count)`-based
loss weights — they bias the GNN toward principles people actually talk
about most.

## Pipeline

```
xlsx  ─►  raw.jsonl  ─►  clean.jsonl  ─►  rule_extracted.jsonl  ─►  final.jsonl
                                                                       │
                                                                       ▼
                                                          dataset.jsonl + summary.json
```

## CLI

Run from this folder, or from the repo root with `PYTHONPATH=.`.

### `python -m tune`

| Flag | Default | Description |
|------|---------|-------------|
| `--in DIR` | required | Directory containing `dm_shorts_*.xlsx` files. |
| `--out DIR` | `../outs/transcripts` | Output directory for all stage artifacts. |
| `--stage {all,ingest,clean,rules,aggregate}` | `all` | Run a single stage or the full pipeline. Useful for re-running just the slow rule extraction without re-ingesting. |

Outputs into `--out`:

| File | Stage | Contents |
|------|-------|----------|
| `raw.jsonl` | ingest | One row per transcript line as parsed from the xlsx. |
| `clean.jsonl` | clean | Same rows after normalization (lowercasing, punctuation, dedup). |
| `rule_extracted.jsonl` | rules | Per-document principle / action / target matches. |
| `final.jsonl` | aggregate | Joined cleaned + extracted rows. |
| `dataset.jsonl` | aggregate | Training-ready records (one per surface form). |
| `summary.json` | aggregate | Aggregate `principle_counts`, `action_counts`, dataset stats. Consumed by `network.cli train --summary-json`. |

### Examples

Full pipeline:

```bash
PYTHONPATH=. python -m tune \
  --in  data/cliff_transcripts \
  --out outs/transcripts
```

Re-run only rule extraction after editing the principle vocabulary:

```bash
PYTHONPATH=. python -m tune \
  --in  data/cliff_transcripts \
  --out outs/transcripts \
  --stage rules
```

## Module layout

- `cli.py` / `__main__.py` — argparse front-end.
- `pipeline.py` — `run_pipeline(input_dir, out_dir, stage)` orchestrator.
- `rule_extract.py` — principle / action / target rule matchers (the heart
  of the weak-supervision step).
- Plus per-stage helpers for ingest / clean / aggregate.

## Downstream consumer

`network.cli train --summary-json outs/transcripts/summary.json` reads the
`principle_counts` map and converts it into per-principle CE weights
`w_p ∝ log(1 + count_p)`. Skip this flag for uniform weighting.
