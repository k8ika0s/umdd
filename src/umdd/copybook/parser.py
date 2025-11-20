"""Minimal copybook parser for synthetic generation.

Supports a subset:
- PIC X(n)
- PIC 9(n) [optionally SIGN LEADING/TRAILING; S prefix]
- COMP-3 (packed decimal) tagged via USAGE COMP-3
- COMP/COMP-5 (binary; default big endian for synthetic)
- REDEFINES/RENAMES not supported (synthetic path only)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PIC_RE = re.compile(r"PIC\s+([XS9V\(\)\.\-]+)", re.IGNORECASE)
USAGE_RE = re.compile(r"USAGE\s+([A-Z0-9\-]+)", re.IGNORECASE)
NAME_RE = re.compile(r"^\s*\d+\s+([A-Z0-9_-]+)", re.IGNORECASE)
OCCURS_RE = re.compile(r"OCCURS\s+(\d+)", re.IGNORECASE)


@dataclass
class Field:
    name: str
    pic: str
    usage: str | None
    occurs: int = 1


def parse_copybook(text: str) -> list[Field]:
    fields: list[Field] = []
    for line in text.splitlines():
        if not line.strip() or line.strip().startswith("*"):
            continue
        name_match = NAME_RE.search(line)
        pic_match = PIC_RE.search(line)
        if not name_match or not pic_match:
            continue
        name = name_match.group(1).upper()
        pic = pic_match.group(1).upper()
        usage_match = USAGE_RE.search(line)
        usage = usage_match.group(1).upper() if usage_match else None
        occurs_match = OCCURS_RE.search(line)
        occurs = int(occurs_match.group(1)) if occurs_match else 1
        fields.append(Field(name=name, pic=pic, usage=usage, occurs=occurs))
    return fields
