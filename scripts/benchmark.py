"""Micro-benchmarks for heuristics on synthetic data."""

from __future__ import annotations

import time
from pathlib import Path

from umdd.data.generator import generate_synthetic_dataset
from umdd.decoder import heuristic_decode


def benchmark_decode(records: int = 1000, runs: int = 3) -> dict[str, float]:
    data, _ = generate_synthetic_dataset(count=records)
    total_bytes = len(data)
    best = None
    for _ in range(runs):
        start = time.perf_counter()
        heuristic_decode(data)
        elapsed = time.perf_counter() - start
        best = elapsed if best is None or elapsed < best else best
    mbps = (total_bytes / 1_000_000) / best if best else 0.0
    return {"records": records, "bytes": total_bytes, "best_seconds": best or 0.0, "mbps": mbps}


if __name__ == "__main__":
    result = benchmark_decode()
    print(result)
