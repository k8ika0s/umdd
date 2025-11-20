"""Multi-head training loop for codepage + token tagging + boundary detection."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from umdd.data.generator import generate_synthetic_dataset, iter_records_with_rdw
from umdd.model import EncoderConfig, MultiHeadModel

TAG_PAD = 0
TAG_RDW = 1
TAG_TEXT = 2
TAG_PACKED = 3
TAG_BINARY = 4
TAG_TO_ID = {
    "PAD": TAG_PAD,
    "RDW": TAG_RDW,
    "TEXT": TAG_TEXT,
    "PACKED": TAG_PACKED,
    "BINARY": TAG_BINARY,
}
ID_TO_TAG = {v: k for k, v in TAG_TO_ID.items()}
BOUNDARY_PAD = -100  # ignored by CE loss


@dataclass
class MultiTaskConfig:
    output_dir: Path
    codepages: tuple[str, ...] = ("cp037", "cp273", "cp500", "cp1047")
    samples_per_codepage: int = 128
    max_len: int = 96
    batch_size: int = 32
    learning_rate: float = 1e-3
    epochs: int = 2
    embed_dim: int = 48
    num_heads: int = 2
    num_layers: int = 1
    ff_dim: int = 96
    dropout: float = 0.1
    device: str = "cpu"
    tag_loss_weight: float = 1.0
    boundary_loss_weight: float = 1.0
    codepage_loss_weight: float = 1.0
    num_workers: int = 0


class SyntheticLabeledDataset(Dataset[dict[str, torch.Tensor]]):
    """RDW synthetic data with byte-level labels for tags + boundaries."""

    def __init__(self, cfg: MultiTaskConfig) -> None:
        self.cfg = cfg
        self.samples: list[dict[str, torch.Tensor]] = []
        for label, cp in enumerate(cfg.codepages):
            data, _meta = generate_synthetic_dataset(count=cfg.samples_per_codepage, codepage=cp)
            for length, body in iter_records_with_rdw(data):
                record_with_rdw = length.to_bytes(2, "big") + b"\x00\x00" + body
                sample = self._build_sample(
                    record_with_rdw, codepage_label=label, body_len=len(body), rdw_len=length
                )
                self.samples.append(sample)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return self.samples[idx]

    def _build_sample(
        self, record: bytes, codepage_label: int, body_len: int, rdw_len: int
    ) -> dict[str, torch.Tensor]:
        max_len = self.cfg.max_len
        tokens = torch.zeros(max_len, dtype=torch.long)
        tag_labels = torch.full((max_len,), TAG_PAD, dtype=torch.long)
        boundary_labels = torch.full((max_len,), BOUNDARY_PAD, dtype=torch.long)

        usable = min(len(record), max_len)
        tokens[:usable] = torch.tensor(list(record[:usable]), dtype=torch.long)

        # Layout from generator: RDW (4 bytes), TEXT, PACKED (4 bytes), BINARY (4 bytes).
        rdw_bytes = 4
        text_len = max(body_len - 8, 0)
        packed_len = 4
        binary_len = 4

        spans = [
            (0, min(rdw_bytes, usable), TAG_RDW),
            (rdw_bytes, min(rdw_bytes + text_len, usable), TAG_TEXT),
            (rdw_bytes + text_len, min(rdw_bytes + text_len + packed_len, usable), TAG_PACKED),
            (
                rdw_bytes + text_len + packed_len,
                min(rdw_bytes + text_len + packed_len + binary_len, usable),
                TAG_BINARY,
            ),
        ]
        for start, end, tag in spans:
            if start >= usable:
                continue
            tag_labels[start:end] = tag
        # Boundary labels: mark each span start.
        for start, _end, _tag in spans:
            if start < usable:
                boundary_labels[start] = 1
        # Non-boundary positions inside usable length.
        for i in range(usable):
            if boundary_labels[i] == BOUNDARY_PAD:
                boundary_labels[i] = 0

        attention_mask = torch.ones(max_len, dtype=torch.bool)
        attention_mask[:usable] = False  # Transformer expects True for padding

        return {
            "tokens": tokens,
            "attention_mask": attention_mask,
            "tag_labels": tag_labels,
            "boundary_labels": boundary_labels,
            "codepage_label": torch.tensor(codepage_label, dtype=torch.long),
        }


def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {k: torch.stack([item[k] for item in batch], dim=0) for k in keys}


def _masked_accuracy(preds: torch.Tensor, labels: torch.Tensor, ignore_index: int) -> float:
    mask = labels != ignore_index
    if mask.sum() == 0:
        return 0.0
    correct = (preds[mask] == labels[mask]).sum().item()
    return correct / mask.sum().item()


def train_multitask(config: MultiTaskConfig) -> dict:
    """Train multi-head model on synthetic labeled data."""
    device = torch.device(config.device)
    ds = SyntheticLabeledDataset(config)
    loader = DataLoader(
        ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        collate_fn=_collate,
    )

    encoder_cfg = EncoderConfig(
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        ff_dim=config.ff_dim,
        dropout=config.dropout,
        max_len=config.max_len,
    )
    model = MultiHeadModel(
        encoder_cfg=encoder_cfg, num_codepages=len(config.codepages), num_tags=len(TAG_TO_ID)
    )
    model.to(device)

    cp_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.CrossEntropyLoss()
    tag_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.CrossEntropyLoss(
        ignore_index=TAG_PAD
    )
    boundary_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.CrossEntropyLoss(
        ignore_index=BOUNDARY_PAD
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    metrics: dict[str, list[float]] = {
        "codepage_acc": [],
        "tag_acc": [],
        "boundary_acc": [],
        "loss": [],
    }

    model.train()
    for _epoch in range(config.epochs):
        for batch in loader:
            tokens = batch["tokens"].to(device)
            attn_mask = batch["attention_mask"].to(device)
            tag_labels = batch["tag_labels"].to(device)
            boundary_labels = batch["boundary_labels"].to(device)
            cp_labels = batch["codepage_label"].to(device)

            out = model(tokens, padding_mask=attn_mask)
            cp_logits = out["codepage_logits"]
            tag_logits = out["tag_logits"]
            boundary_logits = out["boundary_logits"]

            cp_loss = cp_loss_fn(cp_logits, cp_labels)
            tag_loss = tag_loss_fn(tag_logits.view(-1, tag_logits.size(-1)), tag_labels.view(-1))
            boundary_loss = boundary_loss_fn(
                boundary_logits.view(-1, boundary_logits.size(-1)), boundary_labels.view(-1)
            )

            loss = (
                config.codepage_loss_weight * cp_loss
                + config.tag_loss_weight * tag_loss
                + config.boundary_loss_weight * boundary_loss
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                metrics["loss"].append(loss.item())
                metrics["codepage_acc"].append(
                    (cp_logits.argmax(dim=1) == cp_labels).float().mean().item()
                )
                tag_preds = tag_logits.argmax(dim=-1)
                boundary_preds = boundary_logits.argmax(dim=-1)
                metrics["tag_acc"].append(_masked_accuracy(tag_preds, tag_labels, TAG_PAD))
                metrics["boundary_acc"].append(
                    _masked_accuracy(boundary_preds, boundary_labels, BOUNDARY_PAD)
                )

    avg_loss = sum(metrics["loss"]) / len(metrics["loss"]) if metrics["loss"] else 0.0
    avg_cp = (
        sum(metrics["codepage_acc"]) / len(metrics["codepage_acc"])
        if metrics["codepage_acc"]
        else 0.0
    )
    avg_tag = sum(metrics["tag_acc"]) / len(metrics["tag_acc"]) if metrics["tag_acc"] else 0.0
    avg_boundary = (
        sum(metrics["boundary_acc"]) / len(metrics["boundary_acc"])
        if metrics["boundary_acc"]
        else 0.0
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = config.output_dir / "multihead.pt"
    encoder_payload = asdict(encoder_cfg)
    config_payload = asdict(config)
    config_payload["output_dir"] = str(config.output_dir)
    torch.save(
        {
            "model_state": model.state_dict(),
            "encoder_cfg": encoder_payload,
            "config": config_payload,
            "tag_to_id": TAG_TO_ID,
            "id_to_tag": ID_TO_TAG,
        },
        ckpt_path,
    )
    metrics_payload = {
        "average_loss": round(avg_loss, 6),
        "codepage_accuracy": round(avg_cp, 4),
        "tag_accuracy": round(avg_tag, 4),
        "boundary_accuracy": round(avg_boundary, 4),
        "epochs": config.epochs,
        "codepages": list(config.codepages),
        "samples": len(ds),
        "checkpoint": str(ckpt_path),
    }
    metrics_path = config.output_dir / "multihead_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2))
    return metrics_payload
