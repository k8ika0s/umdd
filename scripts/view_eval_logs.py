"""Quick viewer for eval logs (CSV or JSONL) without extra dependencies.

Shows aggregate printable ratio, record totals, and codepage counts in a Rich table.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Tuple

from rich.console import Console
from rich.table import Table

from umdd.eval.summarize import summarize_log


def _iter_entries(path: Path) -> Iterable[Dict[str, object]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            yield from reader
    else:
        with path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def _tag_from_entry(entry: Dict[str, object]) -> Tuple[str, float]:
    tag = ""
    ratio = 0.0
    if "tag" in entry:
        tag = str(entry["tag"] or "")
    if "average_printable_ratio" in entry:
        ratio = float(entry["average_printable_ratio"])
    elif "evaluation" in entry and isinstance(entry["evaluation"], dict):
        ratio = float(entry["evaluation"].get("average_printable_ratio", 0.0))
        tag = tag or str(entry.get("tag", ""))
    return tag, ratio


def main() -> None:
    parser = argparse.ArgumentParser(description="View eval logs.")
    parser.add_argument("log", type=Path, help="CSV or JSONL log file.")
    args = parser.parse_args()

    console = Console()
    summary = summarize_log(args.log)

    # Overall summary
    console.print("[bold]Aggregate[/]")
    console.print(
        f"- entries: {summary['entries']}, records: {summary['records_total']}, "
        f"avg printable ratio: {summary['average_printable_ratio']}"
    )

    # Codepage table
    cp_table = Table(title="Codepage Counts")
    cp_table.add_column("Codepage")
    cp_table.add_column("Count", justify="right")
    cp_counts = summary.get("codepage_counts", {}) or {}
    for cp, count in sorted(cp_counts.items(), key=lambda kv: kv[1], reverse=True):
        cp_table.add_row(cp, str(count))
    console.print(cp_table)

    # Ratios by tag (if present)
    tag_counts: Counter[str] = Counter()
    tag_ratio_sum: Counter[str] = Counter()
    for entry in _iter_entries(args.log):
        tag, ratio = _tag_from_entry(entry)
        if tag:
            tag_counts[tag] += 1
            tag_ratio_sum[tag] += ratio
    if tag_counts:
        tag_table = Table(title="Tags")
        tag_table.add_column("Tag")
        tag_table.add_column("Entries", justify="right")
        tag_table.add_column("Avg Ratio", justify="right")
        for tag, count in tag_counts.most_common():
            avg = tag_ratio_sum[tag] / count if count else 0.0
            tag_table.add_row(tag, str(count), f"{avg:.4f}")
        console.print(tag_table)


if __name__ == "__main__":
    main()
