"""Synthetic record generator driven by copybook-like Field specs."""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from umdd.copybook.parser import Field
from umdd.data.generator import (
    DEFAULT_CODEPAGE,
    _rdw_prefix,
    to_packed_decimal,
)


@dataclass
class CopybookSynthConfig:
    codepage: str = DEFAULT_CODEPAGE
    seed: int = 1234
    bdw: bool = False  # if True, wrap records in BDW segments
    max_block_bytes: int = 32_000
    binary_endian: Literal["big", "little"] = "big"
    include_overpunch: bool = True  # add zoned overpunch variants for PIC 9


def _encode_text(value: str, codepage: str, length: int) -> bytes:
    return value.ljust(length)[:length].encode(codepage, errors="replace")


def _encode_numeric(field: Field, rng: random.Random, cfg: CopybookSynthConfig) -> bytes:
    # simple numeric ranges for synthetic data
    width = int(extract_width(field.pic))
    usage = field.usage or ""
    if "COMP-3" in usage:
        num = rng.randint(-(10 ** (width - 1)), 10 ** (width - 1))
        return to_packed_decimal(num, digits=width)
    if "COMP-5" in usage or "COMP" in usage:
        num = rng.randint(-(2**15), 2**15)
        endian: Literal["big", "little"] = "big" if cfg.binary_endian != "little" else "little"
        return num.to_bytes((width + 1) // 2, byteorder=endian, signed=True).rjust(width, b"\x00")
    # zoned decimal with optional overpunch
    num = rng.randint(-(10 ** (width - 1)), 10 ** (width - 1))
    digits = f"{abs(num):0{width}d}"[-width:]
    if cfg.include_overpunch and digits:
        last = digits[-1]
        overpunch_map_pos = {
            "0": "{",
            "1": "A",
            "2": "B",
            "3": "C",
            "4": "D",
            "5": "E",
            "6": "F",
            "7": "G",
            "8": "H",
            "9": "I",
        }
        overpunch_map_neg = {
            "0": "}",
            "1": "J",
            "2": "K",
            "3": "L",
            "4": "M",
            "5": "N",
            "6": "O",
            "7": "P",
            "8": "Q",
            "9": "R",
        }
        if num >= 0:
            last = overpunch_map_pos.get(last, last)
        else:
            last = overpunch_map_neg.get(last, last)
        digits = digits[:-1] + last
    text = digits
    return text.encode(DEFAULT_CODEPAGE, errors="replace")


def extract_width(pic: str) -> int:
    # PIC X(10), PIC 9(5), PIC 9(3)V99, etc.
    if "(" in pic and ")" in pic:
        inside = pic.split("(", 1)[1].split(")", 1)[0]
        try:
            return int(inside)
        except ValueError:
            return len(pic)
    # crude fallback
    return sum(1 for ch in pic if ch in ("X", "9"))


def synthesize_from_copybook(
    fields: Sequence[Field],
    count: int = 10,
    config: CopybookSynthConfig | None = None,
    include_metadata: bool = True,
) -> tuple[bytes, list[dict]]:
    cfg = config or CopybookSynthConfig()
    rng = random.Random(cfg.seed)
    records: list[bytes] = []
    meta: list[dict] = []

    def build_record(i: int) -> tuple[bytes, dict]:
        parts: list[bytes] = []
        record_meta: dict = {"fields": {}}
        for field in fields:
            for j in range(field.occurs):
                suffix = f"_{j}" if field.occurs > 1 else ""
                fname = field.name + suffix
                if field.pic.startswith("X"):
                    val = f"{fname[:4]}{i:05d}"
                    encoded = _encode_text(val, cfg.codepage, extract_width(field.pic))
                elif field.pic.startswith("9"):
                    encoded = _encode_numeric(field, rng, cfg)
                    # keep an approximate string value for metadata
                    record_meta["fields"][fname] = (
                        int.from_bytes(encoded, "big", signed=True)
                        if "COMP" in (field.usage or "")
                        else None
                    )
                else:
                    encoded = _encode_text("?", cfg.codepage, extract_width(field.pic))
                parts.append(encoded)
        body = b"".join(parts)
        return body, record_meta

    for i in range(count):
        body, rmeta = build_record(i)
        if include_metadata:
            meta.append(rmeta)
        records.append(_rdw_prefix(body) + body)

    dataset = b"".join(records)
    if cfg.bdw:
        blocks: list[bytes] = []
        idx = 0
        while idx < len(dataset):
            chunk = dataset[idx : idx + cfg.max_block_bytes]
            bdw = len(chunk).to_bytes(4, "big")
            blocks.append(bdw + chunk)
            idx += cfg.max_block_bytes
        dataset = b"".join(blocks)
    return dataset, meta
