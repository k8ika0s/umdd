"""Helpers to log evaluation summaries for trend tracking."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from umdd.eval.harness import EvalSummary


def summary_to_row(
    summary: EvalSummary | Mapping[str, object], source: str, tag: str | None = None
) -> dict:
    """Flatten EvalSummary into a CSV/JSONL-friendly row."""
    if isinstance(summary, Mapping):
        records_obj: Any = summary.get("records", 0)
        ratio_obj: Any = summary.get("average_printable_ratio", 0.0)
        counts_obj: Any = summary.get("codepage_counts", {})
        notes_obj: Any = summary.get("notes", "")
        records = int(cast(Any, records_obj) or 0)
        average_printable_ratio = float(cast(Any, ratio_obj) or 0.0)
        counts: dict[str, Any]
        if isinstance(counts_obj, Mapping):
            counts = dict(counts_obj)
        elif isinstance(counts_obj, dict):
            counts = counts_obj
        else:
            counts = {}
        notes = str(notes_obj)
    else:
        records = summary.records
        average_printable_ratio = summary.average_printable_ratio
        counts = summary.codepage_counts
        notes = summary.notes
    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "source": source,
        "tag": tag or "",
        "records": records,
        "average_printable_ratio": average_printable_ratio,
        "codepage_counts": json.dumps(counts),
        "notes": notes,
    }


def append_csv(path: Path, row: dict) -> None:
    """Append a row to a CSV file, writing headers when the file is new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def append_jsonl(path: Path, payload: dict) -> None:
    """Append a JSON line (UTF-8) to a log file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
