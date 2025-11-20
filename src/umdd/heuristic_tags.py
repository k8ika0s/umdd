"""Heuristic field tagging for baselines.

Rules (lightweight, non-ML):
- RDW span (bytes 0-4) when present, and derive record length.
- Packed decimal bytes (low nibble C/D/F).
- Text-ish spans: sequences of bytes that decode cleanly under CP037 and are printable.
- Binary-ish spans: runs of low-printability bytes outside text spans.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Span:
    start: int
    end: int
    tag: str


def tag_fields(data: bytes) -> list[Span]:
    """Detect RDW, packed decimal nibbles, and coarse text/binary spans."""
    spans: list[Span] = []
    idx = 0
    total = len(data)
    # RDW detection (first 4 bytes) only if length is sane.
    if total >= 4:
        rdw_len = int.from_bytes(data[0:2], "big")
        if 4 <= rdw_len <= total:
            spans.append(Span(0, 4, "RDW"))
            spans.append(Span(4, rdw_len, "RECORD_BODY"))
            idx = 4

    # text-ish spans using CP037 decoding
    def emit_span(start: int, end: int, tag: str) -> None:
        if start < end:
            spans.append(Span(start, end, tag))

    text_start = None
    printable = set(
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789 .,;:-_+/&'()[]{}"
    )

    # simple scan: detect packed-decimal bytes (0x0C/0x0D/0x0F nibble) heuristically
    while idx < total:
        byte = data[idx]
        low = byte & 0x0F
        if low in (0x0C, 0x0D, 0x0F):
            spans.append(Span(idx, idx + 1, "PACKED_DECIMAL"))
        is_printable = byte in printable
        if is_printable:
            if text_start is None:
                text_start = idx
        else:
            if text_start is not None:
                emit_span(text_start, idx, "TEXT")
                text_start = None
        idx += 1

    if text_start is not None:
        emit_span(text_start, total, "TEXT")

    # derive binary spans as gaps between text spans (excluding RDW and packed tags)
    # If no binary span was found but we have non-text bytes, mark them.
    if not any(span.tag == "BINARY" for span in spans):
        covered = [False] * total
        for span in spans:
            if span.tag in {"RDW", "RECORD_BODY"}:
                continue
            for i in range(span.start, min(span.end, total)):
                covered[i] = True
        start = 0
        while start < total:
            if not covered[start]:
                end = start
                while end < total and not covered[end]:
                    end += 1
                emit_span(start, end, "BINARY")
                start = end
            else:
                start += 1
    return spans
