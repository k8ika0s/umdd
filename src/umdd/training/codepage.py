"""Codepage detection head: synthetic data and minimal train loop.

Rationales:
- Lock in interfaces early (config, dataset, trainer) so scripts/CI can call them
  even before full data pipelines are ready.
- Use synthetic RDW-prefixed records from the generator to provide a reproducible
  training corpus and avoid blocking on real datasets.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from umdd.data.generator import DEFAULT_CODEPAGE, generate_synthetic_dataset, iter_records_with_rdw


@dataclass
class CodepageTrainingConfig:
    output_dir: Path
    codepages: Sequence[str] = (DEFAULT_CODEPAGE, "cp273", "cp500", "cp1047")
    records_per_page: int = 128
    batch_size: int = 64
    learning_rate: float = 1e-3
    epochs: int = 2
    max_len: int = 128
    embed_dim: int = 32
    num_workers: int = 0
    device: str = "cpu"
    dataset_paths: dict[str, list[Path]] | None = (
        None  # optional external RDW datasets per codepage
    )
    synthetic_extra_per_codepage: int = 0  # add-on synthetic records when real data is present


class CodepageDataset(Dataset[tuple[torch.Tensor, int]]):
    """Dataset of RDW record bodies paired with codepage labels (external or synthetic)."""

    def __init__(
        self,
        codepages: Sequence[str],
        records_per_page: int,
        max_len: int = 128,
        dataset_paths: dict[str, list[Path]] | None = None,
        synthetic_extra_per_codepage: int = 0,
    ) -> None:
        self.samples: list[tuple[torch.Tensor, int]] = []
        self.codepages = list(codepages)
        self.max_len = max_len
        self.synthetic_extra_per_codepage = synthetic_extra_per_codepage
        self.dataset_paths: dict[str, list[Path]] = {}
        for k, paths in (dataset_paths or {}).items():
            self.dataset_paths[k] = [Path(p) for p in paths]

        for label, page in enumerate(self.codepages):
            real_paths = self.dataset_paths.get(page, [])
            if real_paths:
                for path in real_paths:
                    data = path.read_bytes()
                    for _length, body in iter_records_with_rdw(data):
                        self.samples.append((self._bytes_to_tensor(body), label))
                if self.synthetic_extra_per_codepage > 0:
                    synth, _meta = generate_synthetic_dataset(
                        count=self.synthetic_extra_per_codepage, codepage=page
                    )
                    for _length, body in iter_records_with_rdw(synth):
                        self.samples.append((self._bytes_to_tensor(body), label))
            else:
                data, _meta = generate_synthetic_dataset(count=records_per_page, codepage=page)
                for _length, body in iter_records_with_rdw(data):
                    self.samples.append((self._bytes_to_tensor(body), label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.samples[idx]

    def _bytes_to_tensor(self, body: bytes) -> torch.Tensor:
        truncated = body[: self.max_len]
        padded = truncated.ljust(self.max_len, b"\x00")
        return torch.tensor(list(padded), dtype=torch.long)


class CodepageClassifier(nn.Module):
    """Tiny embedding + mean-pool classifier."""

    def __init__(self, embed_dim: int, num_classes: int, vocab_size: int = 256) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        pooled = embedded.mean(dim=1)  # (batch, embed_dim)
        return self.classifier(pooled)


def _accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)


def train_codepage_model(config: CodepageTrainingConfig) -> dict:
    """Train the heuristic codepage classifier on synthetic data and save a checkpoint."""
    device = torch.device(config.device)
    ds = CodepageDataset(
        config.codepages,
        config.records_per_page,
        max_len=config.max_len,
        dataset_paths=config.dataset_paths,
        synthetic_extra_per_codepage=config.synthetic_extra_per_codepage,
    )
    loader = DataLoader(
        ds, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers
    )

    model = CodepageClassifier(embed_dim=config.embed_dim, num_classes=len(config.codepages))
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    losses: list[float] = []
    accs: list[float] = []

    model.train()
    for _epoch in range(config.epochs):
        for _batch, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            batch_acc = _accuracy(logits, y)
            losses.append(loss.item())
            accs.append(batch_acc)

    avg_loss = sum(losses) / len(losses) if losses else 0.0
    avg_acc = sum(accs) / len(accs) if accs else 0.0

    config.output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = config.output_dir / "codepage_head.pt"
    torch.save({"model_state": model.state_dict(), "config": asdict(config)}, ckpt_path)

    metrics = {
        "average_loss": round(avg_loss, 6),
        "average_accuracy": round(avg_acc, 4),
        "batches": len(losses),
        "epochs": config.epochs,
        "codepages": list(config.codepages),
        "checkpoint": str(ckpt_path),
    }
    metrics_path = config.output_dir / "codepage_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    return metrics
