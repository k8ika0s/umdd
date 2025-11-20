"""Lightweight multi-head byte model for codepage/field/boundary inference."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn


def _build_positional_encoding(max_len: int, dim: int) -> torch.Tensor:
    """Sinusoidal positional encoding (batch dimension will be broadcast)."""
    pe = torch.zeros(max_len, dim)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10_000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, max_len, dim)


def _masked_mean(encoded: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
    """Mean-pool over non-padded positions."""
    if padding_mask is None:
        return encoded.mean(dim=1)
    # padding_mask: True where padded
    keep = (~padding_mask).float().unsqueeze(-1)  # (batch, seq, 1)
    summed = (encoded * keep).sum(dim=1)
    denom = keep.sum(dim=1).clamp(min=1.0)
    return summed / denom


@dataclass
class EncoderConfig:
    vocab_size: int = 256
    embed_dim: int = 64
    num_heads: int = 2
    num_layers: int = 1
    ff_dim: int = 128
    dropout: float = 0.1
    max_len: int = 128


class ByteEncoder(nn.Module):
    """Byte-level encoder with embeddings + Transformer encoder."""

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.num_layers)
        self.positional_encoding: torch.Tensor
        pe = _build_positional_encoding(cfg.max_len, cfg.embed_dim)
        self.register_buffer("positional_encoding", pe, persistent=False)

    def forward(
        self, tokens: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # tokens: (batch, seq)
        x = self.embedding(tokens)
        x = x + self.positional_encoding[:, : x.size(1), :]
        return self.encoder(x, src_key_padding_mask=padding_mask)


class MultiHeadModel(nn.Module):
    """Shared encoder with codepage + tag + boundary heads."""

    def __init__(
        self,
        encoder_cfg: EncoderConfig,
        num_codepages: int,
        num_tags: int,
    ) -> None:
        super().__init__()
        self.encoder = ByteEncoder(encoder_cfg)
        self.codepage_head = nn.Linear(encoder_cfg.embed_dim, num_codepages)
        self.tag_head = nn.Linear(encoder_cfg.embed_dim, num_tags)
        self.boundary_head = nn.Linear(encoder_cfg.embed_dim, 2)

    def forward(
        self, tokens: torch.Tensor, padding_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        encoded = self.encoder(tokens, padding_mask=padding_mask)
        pooled = _masked_mean(encoded, padding_mask)
        return {
            "encoded": encoded,
            "codepage_logits": self.codepage_head(pooled),
            "tag_logits": self.tag_head(encoded),
            "boundary_logits": self.boundary_head(encoded),
        }
