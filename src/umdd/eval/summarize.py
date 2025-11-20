"""Summaries over eval logs (CSV or JSONL)."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def _iter_csv(path: Path) -> Iterator[dict[str, str]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        yield from reader


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def summarize_log(path: Path) -> dict[str, object]:
    """Compute simple aggregates from a CSV/JSONL log."""
    ratios: list[float] = []
    records_total = 0
    codepage_counts: dict[str, int] = {}

    iterator = _iter_csv(path) if path.suffix.lower() == ".csv" else _iter_jsonl(path)

    for entry in iterator:
        # CSV uses string values; JSONL may be nested payloads.
        if "average_printable_ratio" in entry:
            ratio_val = entry["average_printable_ratio"]
            ratios.append(float(ratio_val))

        if "records" in entry:
            records_total += int(entry["records"])

        if "codepage_counts" in entry:
            counts_raw = entry["codepage_counts"]
            if isinstance(counts_raw, str):
                counts = json.loads(counts_raw)
            else:
                counts = counts_raw
            for k, v in counts.items():
                codepage_counts[k] = codepage_counts.get(k, 0) + int(v)

    avg_ratio = sum(ratios) / len(ratios) if ratios else 0.0
    return {
        "entries": len(ratios),
        "records_total": records_total,
        "average_printable_ratio": round(avg_ratio, 4),
        "codepage_counts": codepage_counts,
    }
