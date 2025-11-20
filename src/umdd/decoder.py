"""Heuristic decoder scaffolding.

Provides a lightweight, rule-based pass to approximate codepage selection and
text rendering before ML heads land. This keeps the CLI useful for inspection
and establishes test baselines.
"""

from __future__ import annotations

import string

CODEPAGE_CANDIDATES: list[str] = ["cp037", "cp273", "cp500"]
PRINTABLE = set(string.printable)


def _printable_ratio(text: str) -> float:
    """Measure how many characters are printable ASCII-ish."""
    if not text:
        return 0.0
    printable = sum(1 for ch in text if ch in PRINTABLE)
    return printable / len(text)


def heuristic_decode(data: bytes, preview_bytes: int = 256) -> dict[str, object]:
    """Pick the codepage with the best printable ratio and return a preview."""
    best_page = None
    best_score = -1.0
    best_text = ""
    best_notes = ""

    for page in CODEPAGE_CANDIDATES:
        try:
            decoded = data.decode(page, errors="ignore")
        except LookupError:
            continue  # skip codepages not installed in the runtime
        score = _printable_ratio(decoded)
        if score > best_score:
            best_score = score
            best_page = page
            best_text = decoded
            best_notes = "heuristic codepage match"

    preview = best_text[:preview_bytes]
    return {
        "detected_codepage": best_page,
        "printable_ratio": round(best_score, 4),
        "preview": preview,
        "notes": best_notes or "heuristic-only; ML heads will replace this path",
    }
