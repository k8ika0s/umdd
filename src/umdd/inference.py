"""Inference helpers for the multi-head model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from umdd.data.generator import iter_records_with_rdw
from umdd.model import EncoderConfig, MultiHeadModel
from umdd.training.multitask import ID_TO_TAG, TAG_PAD


@dataclass
class InferenceResult:
    record_index: int
    byte_length: int
    codepage: str
    codepage_confidence: float
    tag_spans: list[dict[str, Any]]
    boundary_positions: list[int]


def load_multihead_checkpoint(path: Path, map_location: str = "cpu") -> tuple[MultiHeadModel, dict]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    encoder_cfg = EncoderConfig(**ckpt["encoder_cfg"])
    config_dict = ckpt.get("config", {})
    codepages = tuple(config_dict.get("codepages", []))
    model = MultiHeadModel(
        encoder_cfg=encoder_cfg, num_codepages=len(codepages), num_tags=len(ckpt["tag_to_id"])
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, {
        "encoder_cfg": encoder_cfg,
        "codepages": codepages,
        "tag_to_id": ckpt["tag_to_id"],
        "id_to_tag": {int(k): v for k, v in ckpt["id_to_tag"].items()},
        "config": config_dict,
    }


def _build_tensor(record: bytes, max_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    usable = min(len(record), max_len)
    tokens = torch.zeros(max_len, dtype=torch.long)
    tokens[:usable] = torch.tensor(list(record[:usable]), dtype=torch.long)
    attn_mask = torch.ones(max_len, dtype=torch.bool)
    attn_mask[:usable] = False
    return tokens, attn_mask


def _tags_to_spans(tags: list[str]) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    if not tags:
        return spans
    start = 0
    current = tags[0]
    for idx, tag in enumerate(tags[1:], start=1):
        if tag != current:
            spans.append({"start": start, "end": idx, "tag": current})
            start = idx
            current = tag
    spans.append({"start": start, "end": len(tags), "tag": current})
    return spans


def infer_bytes(
    data: bytes, checkpoint: Path, max_records: int | None = None, device: str = "cpu"
) -> list[InferenceResult]:
    model, meta = load_multihead_checkpoint(checkpoint, map_location=device)
    codepages = meta["codepages"]
    encoder_cfg: EncoderConfig = meta["encoder_cfg"]
    id_to_tag = meta["id_to_tag"] or ID_TO_TAG

    results: list[InferenceResult] = []
    for idx, (length, body) in enumerate(iter_records_with_rdw(data)):
        if max_records is not None and idx >= max_records:
            break
        record = length.to_bytes(2, "big") + b"\x00\x00" + body
        tokens, attn_mask = _build_tensor(record, encoder_cfg.max_len)

        with torch.no_grad():
            out = model(tokens.unsqueeze(0), padding_mask=attn_mask.unsqueeze(0))
            cp_probs = F.softmax(out["codepage_logits"], dim=-1)[0]
            cp_id = int(cp_probs.argmax().item())
            cp_conf = float(cp_probs.max().item())

            tag_ids = out["tag_logits"].argmax(dim=-1)[0].tolist()
            usable = min(len(record), encoder_cfg.max_len)
            tags = [id_to_tag.get(tid, "PAD") for tid in tag_ids[:usable] if tid != TAG_PAD]
            spans = _tags_to_spans(tags)

            boundary_preds = out["boundary_logits"].argmax(dim=-1)[0].tolist()
            boundary_positions = [i for i, val in enumerate(boundary_preds[:usable]) if val == 1]

        results.append(
            InferenceResult(
                record_index=idx,
                byte_length=len(record),
                codepage=codepages[cp_id] if codepages else f"id-{cp_id}",
                codepage_confidence=cp_conf,
                tag_spans=spans,
                boundary_positions=boundary_positions,
            )
        )
    return results
