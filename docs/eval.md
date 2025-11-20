# Evaluation Harness (Heuristic Baseline)

## Purpose
- Provide a reproducible checkpoint for decoder behavior before ML heads arrive.
- Use synthetic RDW datasets to measure codepage guesses and printable ratios.
- Offer a stable interface so future model-based evaluators can drop in without changing callers.

## What runs today
- Heuristic decoder over CP037/273/500/1047 candidates, scoring printable ASCII-ish output.
- Dataset splitting via RDW to evaluate per-record rather than whole-file blobs.
- Summary stats: record count, average printable ratio, codepage frequency, and sample previews.

## How to use
- CLI: `umdd eval heuristic --input data.bin` to run on a real/supplied file.
- Synthetic: omit `--input` and optionally set `--count`/`--seed` to generate and evaluate synthetic records in one command.
- JSON output is available via `--output report.json`.
- Logging: `--log-csv logs/eval.csv --log-jsonl logs/eval.jsonl` appends summaries for trend tracking; if `--output` is set, logs default to `logs/eval.csv`/`logs/eval.jsonl` unless `--no-auto-log` is used. Summarize with `umdd eval summarize logs/eval.csv`.
- Viewer: `python scripts/view_eval_logs.py logs/eval.csv` to print aggregates/codepage counts via Rich.
- Benchmark: `python scripts/benchmark.py` to track heuristic throughput on synthetic data.

## Why this exists
- Baseline now: keeps the project usable for inspection and regression testing before models ship.
- Future hook: the same interface and summary format can be backed by model heads later, preserving scripts/automation built today.

## CI logging idea
- Run `umdd eval heuristic --output logs/eval/latest.json --log-csv logs/eval.csv --log-jsonl logs/eval.jsonl` in CI on synthetic fixtures to track drift. Collect logs as artifacts for trend review.
