"""Lightweight evaluation harness for heuristic decoders.

Purpose:
- Provide a reproducible baseline before ML heads land.
- Run on synthetic RDW datasets and summarize codepage guesses + printable ratios.
- Serve as a scaffolding point for future model-based evaluators.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from umdd.data.generator import DEFAULT_CODEPAGE, generate_synthetic_dataset, iter_records_with_rdw
from umdd.decoder import heuristic_decode


@dataclass
class RecordEval:
    detected_codepage: str
    printable_ratio: float
    preview: str


@dataclass
class EvalSummary:
    records: int
    average_printable_ratio: float
    codepage_counts: dict[str, int]
    samples: list[RecordEval]
    notes: str


def evaluate_dataset(data: bytes, preview_bytes: int = 128, sample_limit: int = 3) -> EvalSummary:
    """Evaluate with the heuristic decoder and summarize results."""
    counts: Counter[str] = Counter()
    ratios: list[float] = []
    samples: list[RecordEval] = []

    for _length, body in iter_records_with_rdw(data):
        result: dict[str, Any] = heuristic_decode(body, preview_bytes=preview_bytes)
        page = str(result.get("detected_codepage") or "unknown")
        ratio = float(result.get("printable_ratio") or 0.0)
        counts[page] += 1
        ratios.append(ratio)

        if len(samples) < sample_limit:
            samples.append(
                RecordEval(
                    detected_codepage=page,
                    printable_ratio=ratio,
                    preview=str(result.get("preview") or ""),
                )
            )

    avg_ratio = sum(ratios) / len(ratios) if ratios else 0.0
    return EvalSummary(
        records=len(ratios),
        average_printable_ratio=round(avg_ratio, 4),
        codepage_counts=dict(counts),
        samples=samples,
        notes="heuristic-only baseline; replace with model heads when available",
    )


def evaluate_synthetic(
    count: int = 8, seed: int = 1234, codepage: str = DEFAULT_CODEPAGE
) -> dict[str, object]:
    """Generate synthetic data and return evaluation plus generator metadata."""
    data, metadata = generate_synthetic_dataset(count=count, seed=seed, codepage=codepage)
    summary = evaluate_dataset(data)
    return {
        "generator": {"count": count, "seed": seed, "codepage": codepage},
        "evaluation": summary,
        "metadata": metadata,
    }
