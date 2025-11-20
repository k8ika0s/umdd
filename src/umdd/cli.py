from collections.abc import Mapping
from pathlib import Path
from typing import Any

import orjson
import typer
from rich.console import Console

from umdd.copybook.parser import parse_copybook
from umdd.copybook.synth import CopybookSynthConfig, synthesize_from_copybook
from umdd.data.generator import generate_synthetic_dataset
from umdd.decoder import heuristic_decode
from umdd.eval.harness import evaluate_dataset, evaluate_synthetic
from umdd.eval.report import append_csv, append_jsonl, summary_to_row
from umdd.eval.summarize import summarize_log
from umdd.inference import infer_bytes
from umdd.training.codepage import CodepageTrainingConfig, train_codepage_model
from umdd.training.multitask import MultiTaskConfig, RealDataSpec, train_multitask

app = typer.Typer(help="Decode mainframe data into structured outputs.")
dataset_app = typer.Typer(help="Dataset helpers (synthetic fixtures, conversions).")
eval_app = typer.Typer(help="Evaluation harness for heuristics and future model heads.")
train_app = typer.Typer(help="Training entrypoints for model heads.")
console = Console()
SUPPORTED_FORMATS = {"json", "csv", "arrow"}

app.add_typer(dataset_app, name="dataset")
app.add_typer(eval_app, name="eval")
app.add_typer(train_app, name="train")


def _read_bytes(path: Path) -> bytes:
    if not path.is_file():
        raise typer.BadParameter(f"Input file not found: {path}")
    return path.read_bytes()


@app.command()
def decode(
    input: Path = typer.Argument(..., help="Raw mainframe dataset to decode."),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Optional path to write structured output."
    ),
    format: str = typer.Option(
        "json", "--format", "-f", help="Output format hint: json | csv | arrow."
    ),
    preview: int = typer.Option(
        256, "--preview-bytes", help="Number of decoded characters to include in preview."
    ),
) -> None:
    """Decode a dataset using a heuristic path until model heads land."""
    fmt = format.lower()
    if fmt not in SUPPORTED_FORMATS:
        raise typer.BadParameter(f"Unsupported format '{format}'. Choose from {SUPPORTED_FORMATS}.")

    data = _read_bytes(input)
    console.print(f"[bold green]Read[/] {len(data)} bytes from {input}")

    heuristic = heuristic_decode(data, preview_bytes=preview)
    payload = {
        "input": str(input),
        "bytes": len(data),
        "requested_format": fmt,
        "note": "ML pipeline pending; emitting heuristic decode metadata.",
        "heuristic": heuristic,
    }

    if output:
        output.write_bytes(orjson.dumps(payload))
        console.print(f"[bold green]Wrote heuristic output[/] to {output}")
    else:
        console.print(orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode())


@app.command()
def infer(
    input: Path = typer.Argument(..., help="Dataset to run model inference against."),
    checkpoint: Path = typer.Option(..., "--checkpoint", "-c", help="Multi-head model checkpoint."),
    max_records: int = typer.Option(1, "--max-records", help="Limit number of records processed."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Optional path to write JSON."),
) -> None:
    """Run the multi-head model (codepage + tag + boundary) on input data."""
    data = _read_bytes(input)
    results = infer_bytes(data, checkpoint=checkpoint, max_records=max_records)
    payload = [
        {
            "record_index": r.record_index,
            "byte_length": r.byte_length,
            "codepage": r.codepage,
            "codepage_confidence": r.codepage_confidence,
            "tag_spans": r.tag_spans,
            "boundary_positions": r.boundary_positions,
        }
        for r in results
    ]
    if output:
        output.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
        console.print(f"[bold green]Wrote inference output[/] to {output}")
    else:
        console.print(orjson.dumps(payload, option=orjson.OPT_INDENT_2).decode())


@dataset_app.command("synthetic")
def dataset_synthetic(
    output: Path = typer.Argument(..., help="Path to write the synthetic dataset (.bin)."),
    metadata: Path | None = typer.Option(
        None, "--metadata", "-m", help="Optional path to write JSON metadata about records."
    ),
    count: int = typer.Option(8, "--count", "-c", help="Number of records to emit."),
    seed: int = typer.Option(1234, "--seed", help="Seed for reproducible generation."),
) -> None:
    """Generate RDW-prefixed synthetic data mixing EBCDIC text, packed decimal, and binary ints."""
    data, meta = generate_synthetic_dataset(count=count, seed=seed)
    output.write_bytes(data)
    console.print(f"[bold green]Wrote[/] {len(data)} bytes to {output} ({count} records).")

    if metadata:
        metadata.write_bytes(orjson.dumps(meta, option=orjson.OPT_INDENT_2))
        console.print(f"[bold green]Wrote metadata[/] to {metadata}")


@dataset_app.command("copybook")
def dataset_copybook(
    copybook: Path = typer.Argument(..., help="Copybook file to drive synthetic generation."),
    output: Path = typer.Argument(..., help="Path to write the synthetic dataset (.bin)."),
    metadata: Path | None = typer.Option(
        None, "--metadata", "-m", help="Optional path to write JSON metadata about records."
    ),
    count: int = typer.Option(8, "--count", "-c", help="Number of records to emit."),
    seed: int = typer.Option(1234, "--seed", help="Seed for reproducible generation."),
    codepage: str = typer.Option("cp037", "--codepage", "-p", help="EBCDIC codepage."),
    bdw: bool = typer.Option(False, "--bdw", help="Wrap output in BDW blocks."),
    max_block_bytes: int = typer.Option(32_000, "--max-block-bytes", help="BDW block size."),
    binary_endian: str = typer.Option("big", "--binary-endian", help="Endianness for COMP."),
    no_overpunch: bool = typer.Option(False, "--no-overpunch", help="Disable zoned overpunch."),
) -> None:
    """Generate synthetic data from a copybook, including packed/binary/zoned fields."""
    fields = parse_copybook(copybook.read_text())
    cfg = CopybookSynthConfig(
        codepage=codepage,
        seed=seed,
        bdw=bdw,
        max_block_bytes=max_block_bytes,
        binary_endian="little" if binary_endian.lower().startswith("l") else "big",
        include_overpunch=not no_overpunch,
    )
    data, meta = synthesize_from_copybook(fields, count=count, config=cfg)
    output.write_bytes(data)
    console.print(
        f"[bold green]Wrote[/] {len(data)} bytes to {output} ({count} records) from copybook."
    )
    if metadata:
        metadata.write_bytes(orjson.dumps(meta, option=orjson.OPT_INDENT_2))
        console.print(f"[bold green]Wrote metadata[/] to {metadata}")


@eval_app.command("heuristic")
def eval_heuristic(
    input: Path | None = typer.Option(
        None,
        "--input",
        "-i",
        help="Dataset to evaluate. If omitted, a synthetic set is generated.",
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Optional path to write evaluation JSON."
    ),
    log_csv: Path | None = typer.Option(
        None, "--log-csv", help="Append summary as a CSV row for trend tracking."
    ),
    log_jsonl: Path | None = typer.Option(
        None, "--log-jsonl", help="Append full payload as JSONL for trend tracking."
    ),
    auto_log: bool = typer.Option(
        True,
        "--auto-log/--no-auto-log",
        help=(
            "If set and --output is provided, append logs to logs/eval.csv and "
            "logs/eval.jsonl by default."
        ),
    ),
    tag: str | None = typer.Option(None, "--tag", help="Optional tag to mark this run."),
    count: int = typer.Option(8, "--count", "-c", help="Records to generate for synthetic eval."),
    seed: int = typer.Option(1234, "--seed", help="Seed for synthetic generation."),
) -> None:
    """Run heuristic evaluation over a dataset or synthetic sample."""
    if auto_log and output:
        log_csv = log_csv or Path("logs/eval.csv")
        log_jsonl = log_jsonl or Path("logs/eval.jsonl")

    if input:
        data = _read_bytes(input)
        summary = evaluate_dataset(data)
        payload = {
            "source": str(input),
            "evaluation": summary,
            "note": "heuristic-only baseline; replace when model heads are ready.",
            "tag": tag,
        }
    else:
        payload = evaluate_synthetic(count=count, seed=seed)
        payload["tag"] = tag

    def _serialize(obj: object) -> object:
        # dataclasses from EvalSummary are not JSON-native; convert via __dict__.
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return obj

    # Optional logging for trend tracking.
    evaluation: object | None = payload.get("evaluation")
    if evaluation and log_csv:
        evaluation_obj: Mapping[str, object] | Any | None = None
        if isinstance(evaluation, Mapping):
            evaluation_obj = evaluation
        elif hasattr(evaluation, "__dict__"):
            evaluation_obj = evaluation

        if evaluation_obj is not None:
            row = summary_to_row(evaluation_obj, source=str(input or "synthetic"), tag=tag)
            append_csv(log_csv, row)
            console.print(f"[bold green]Appended CSV log[/] to {log_csv}")

    if log_jsonl:
        append_jsonl(log_jsonl, payload)
        console.print(f"[bold green]Appended JSONL log[/] to {log_jsonl}")

    if output:
        output.write_bytes(orjson.dumps(payload, default=_serialize))
        console.print(f"[bold green]Wrote evaluation report[/] to {output}")
    else:
        console.print(
            orjson.dumps(payload, option=orjson.OPT_INDENT_2, default=_serialize).decode()
        )


@eval_app.command("summarize")
def eval_summarize(
    log: Path = typer.Argument(..., help="CSV or JSONL log file produced by eval."),
) -> None:
    """Summarize log(s) produced by eval logging."""
    summary = summarize_log(log)
    console.print(orjson.dumps(summary, option=orjson.OPT_INDENT_2).decode())


@train_app.command("codepage")
def train_codepage(
    output_dir: Path = typer.Option(
        Path("artifacts/codepage"), "--output-dir", "-o", help="Where to store checkpoints/metrics."
    ),
    epochs: int = typer.Option(2, "--epochs", help="Training epochs over synthetic data."),
    records_per_page: int = typer.Option(
        128, "--records-per-page", help="Synthetic records per codepage."
    ),
    batch_size: int = typer.Option(64, "--batch-size", help="Batch size."),
    device: str = typer.Option("cpu", "--device", help="Torch device, e.g., cpu or cuda."),
    dataset: list[str] | None = typer.Option(
        None,
        "--dataset",
        "-d",
        help=(
            "Optional mapping CODEPAGE=PATH[,PATH...] to use real RDW data for that "
            "label (repeatable)."
        ),
    ),
    synthetic_extra_per_codepage: int = typer.Option(
        0,
        "--synthetic-extra-per-codepage",
        help=(
            "Add this many synthetic records per codepage even when real data is provided "
            "(data smoothing)."
        ),
    ),
) -> None:
    """Train the codepage head on synthetic data and emit a checkpoint + metrics."""
    dataset_paths: dict[str, list[Path]] = {}
    if dataset:
        for entry in dataset:
            if "=" not in entry:
                raise typer.BadParameter("Dataset mapping must be CODEPAGE=PATH")
            codepage, path_str = entry.split("=", 1)
            paths = [p.strip() for p in path_str.split(",") if p.strip()]
            if not paths:
                raise typer.BadParameter("Dataset mapping must include at least one path")
            dataset_paths[codepage.strip()] = [Path(p).expanduser() for p in paths]

    config = CodepageTrainingConfig(
        output_dir=output_dir,
        epochs=epochs,
        records_per_page=records_per_page,
        batch_size=batch_size,
        device=device,
        dataset_paths=dataset_paths or None,
        synthetic_extra_per_codepage=synthetic_extra_per_codepage,
    )
    console.print("[yellow]Training codepage head on synthetic data (heuristic baseline).[/]")
    metrics = train_codepage_model(config)
    console.print(f"[bold green]Saved checkpoint[/] to {metrics['checkpoint']}")
    console.print(orjson.dumps(metrics, option=orjson.OPT_INDENT_2).decode())


@train_app.command("multitask")
def train_multitask_cli(
    output_dir: Path = typer.Option(
        Path("artifacts/multihead"),
        "--output-dir",
        "-o",
        help="Where to store checkpoints/metrics.",
    ),
    samples_per_codepage: int = typer.Option(
        128, "--samples-per-codepage", help="Synthetic samples per codepage."
    ),
    max_len: int = typer.Option(96, "--max-len", help="Max sequence length."),
    batch_size: int = typer.Option(32, "--batch-size", help="Batch size."),
    epochs: int = typer.Option(2, "--epochs", help="Epochs to train."),
    device: str = typer.Option("cpu", "--device", help="Torch device, e.g., cpu or cuda."),
    real: list[str] | None = typer.Option(
        None,
        "--real",
        help=(
            "Optional real data mapping CODEPAGE=PATH[:copybook[:bdw]] (repeatable). "
            "Use :bdw suffix to indicate BDW blocks; include copybook to derive tags/boundaries."
        ),
    ),
) -> None:
    """Train multi-head model (codepage + tag + boundary) on synthetic data."""
    real_specs: list[tuple[str, Path, Path | None, bool]] = []
    if real:
        for entry in real:
            if "=" not in entry:
                raise typer.BadParameter("Real dataset must be CODEPAGE=PATH[:copybook[:bdw]]")
            codepage, rest = entry.split("=", 1)
            parts = rest.split(":")
            path_str = parts[0]
            copybook = Path(parts[1]).expanduser() if len(parts) >= 2 and parts[1] else None
            bdw = len(parts) >= 3 and parts[2].lower() == "bdw"
            real_specs.append((codepage.strip(), Path(path_str).expanduser(), copybook, bdw))
    config = MultiTaskConfig(
        output_dir=output_dir,
        samples_per_codepage=samples_per_codepage,
        max_len=max_len,
        batch_size=batch_size,
        epochs=epochs,
        device=device,
        real_datasets=[
            RealDataSpec(codepage=cp, path=path, copybook=cb, bdw=bdw)
            for cp, path, cb, bdw in real_specs
        ]
        if real_specs
        else None,
    )
    console.print("[yellow]Training multi-head model on synthetic data.[/]")
    metrics = train_multitask(config)
    console.print(f"[bold green]Saved checkpoint[/] to {metrics['checkpoint']}")
    console.print(orjson.dumps(metrics, option=orjson.OPT_INDENT_2).decode())


if __name__ == "__main__":
    app()
