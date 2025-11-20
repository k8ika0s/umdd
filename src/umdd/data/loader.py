"""Helpers for consuming real RDW/BDW datasets and optional copybook labels."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from umdd.copybook.parser import Field
from umdd.copybook.synth import extract_width
from umdd.data.generator import _rdw_prefix, iter_records_with_rdw


def iter_bdw_records(data: bytes) -> Iterable[tuple[int, bytes]]:
    """Iterate BDW-wrapped VB/VB datasets: each block has a 4-byte length prefix."""
    idx = 0
    total = len(data)
    while idx + 4 <= total:
        block_len = int.from_bytes(data[idx : idx + 4], "big")
        if block_len < 4 or idx + block_len > total:
            break
        block = data[idx + 4 : idx + block_len]
        yield from iter_records_with_rdw(block)
        idx += block_len


def load_records(path: Path, bdw: bool = False) -> list[tuple[int, bytes]]:
    """Load RDW or BDW+RDW records from disk."""
    data = path.read_bytes()
    iterator = iter_bdw_records(data) if bdw else iter_records_with_rdw(data)
    return list(iterator)


@dataclass
class CopybookLabels:
    tag_ids: list[int]
    boundary_ids: list[int]


def labels_from_copybook(
    fields: list[Field],
    body: bytes,
    max_len: int,
    tag_to_id: dict[str, int],
    boundary_pad: int,
) -> CopybookLabels:
    """Derive token tags/boundaries from copybook layout."""
    tags = [tag_to_id["PAD"]] * max_len
    boundaries = [boundary_pad] * max_len

    rdw_len = 4
    usable = min(len(body) + rdw_len, max_len)
    # include RDW span
    tags[0:rdw_len] = [tag_to_id["RDW"]] * min(rdw_len, usable)
    boundaries[0] = 1  # RDW start

    pos = rdw_len
    for field in fields:
        width = extract_width(field.pic)
        for _ in range(field.occurs):
            start = pos
            end = min(pos + width, max_len)
            if start >= usable:
                break
            # map usage -> tag
            usage = (field.usage or field.pic or "").upper()
            if "COMP-3" in usage:
                tag = tag_to_id.get("PACKED", tag_to_id["BINARY"])
            elif "COMP" in usage:
                tag = tag_to_id.get("BINARY", tag_to_id["PACKED"])
            elif field.pic.startswith("9"):
                tag = tag_to_id.get("BINARY", tag_to_id["PACKED"])
            else:
                tag = tag_to_id.get("TEXT", tag_to_id["BINARY"])
            tags[start:end] = [tag] * (end - start)
            boundaries[start] = 1
            pos += width
    # fill remaining non-boundary positions (inside usable) with 0
    for i in range(usable):
        if boundaries[i] == boundary_pad:
            boundaries[i] = 0
    return CopybookLabels(tag_ids=tags, boundary_ids=boundaries)


def rdw_record_with_header(body: bytes) -> bytes:
    """Prefix a record body with RDW bytes for unified tokenization."""
    return _rdw_prefix(body) + body
