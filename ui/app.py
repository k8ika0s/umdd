import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from umdd.inference import InferenceResult, infer_bytes, results_to_arrow, results_to_jsonl
from umdd.manifest import load_manifest, validate_manifest
from umdd.training.multitask import MultiTaskConfig, train_multitask
from umdd.eval.harness import evaluate_dataset


def _render_spans(res: InferenceResult) -> str:
    parts = []
    for span in res.tag_spans:
        parts.append(f"{span['tag']}[{span['start']}-{span['end']}]")
    return ", ".join(parts)


def main() -> None:
    st.title("UMDD Inference, Validation, and Training")
    st.caption(
        "Upload a manifest to validate, run inference with confidences and downloads, or kick off a tiny synthetic training run."
    )
    st.info(
        "Tip: use the notebook or CLI for full control; this UI is a quick way to validate and demo flows with small files."
    )
    st.session_state.setdefault("infer_logs", [])
    st.session_state.setdefault("train_logs", [])
    st.session_state.setdefault("manifest_paths", [])

    tabs = st.tabs(["Validate", "Infer", "Train"])

    with tabs[0]:
        st.subheader("Validate manifest")
        st.write("Manifest describes dataset path, codepage, BDW flag, copybook, and optional checks.")
        manifest_file = st.file_uploader("Upload manifest (json/yaml)", type=["json", "yml", "yaml"])
        if manifest_file:
            tmp_path = Path("uploaded_manifest")
            tmp_path.write_bytes(manifest_file.read())
            mf = load_manifest(tmp_path)
            result = validate_manifest(mf)
            st.json(result)
            st.session_state["manifest_paths"].append(str(mf.path))
            # Optional quick eval on the dataset for heuristics/printability
            if mf.path.exists():
                data = mf.path.read_bytes()
                summary = evaluate_dataset(data)
                st.markdown("**Heuristic eval summary**")
                st.json(summary.__dict__ if hasattr(summary, "__dict__") else summary)
            else:
                st.warning("Manifest file path does not exist; validation incomplete.")
        if st.session_state["manifest_paths"]:
            st.markdown("Recent manifest dataset paths:")
            for p in st.session_state["manifest_paths"][-5:]:
                st.code(p)

    with tabs[1]:
        st.subheader("Inference")
        st.write("Upload a small RDW/BDW sample to preview codepage guesses, spans, boundaries, and confidences.")
        checkpoint = st.text_input("Checkpoint path", "artifacts/multihead/multihead.pt")
        max_records = st.number_input("Max records", min_value=1, value=2, step=1, key="infer_max_records")
        include_conf = st.checkbox("Include confidences", value=True, key="infer_conf")
        fmt = st.selectbox("Output format", ["json", "jsonl", "arrow"], key="infer_fmt")
        log_perf = st.checkbox("Log performance (throughput/latency)", value=True, key="infer_perf")

        uploaded = st.file_uploader("Upload RDW/BDW file", type=["bin", "dat"], key="infer_file")
        if uploaded:
            data = uploaded.read()
            if len(data) > 5_000_000:
                st.warning("Uploaded file is large (>5MB). UI is intended for small samples; use CLI for big runs.")
            st.info(f"Received {len(data)} bytes")
            start = time.time()
            results = infer_bytes(
                data,
                Path(checkpoint),
                max_records=int(max_records),
                include_confidence=include_conf,
            )
            elapsed = time.time() - start
            if log_perf:
                bytes_read = len(data)
                recs = len(results)
                st.success(
                    f"Inference: {elapsed:.3f}s | records: {recs} | throughput: {bytes_read / max(elapsed, 1e-6):.1f} B/s"
                )
            table = pd.DataFrame(
                [
                    {
                        "record_index": r.record_index,
                        "byte_length": r.byte_length,
                        "codepage": r.codepage,
                        "codepage_confidence": r.codepage_confidence,
                        "tag_spans": _render_spans(r),
                        "boundary_positions": r.boundary_positions,
                    }
                    for r in results
                ]
            )
            st.dataframe(table, height=300)

            # Quick codepage distribution visualization
            codepage_counts = table["codepage"].value_counts().rename_axis("codepage").reset_index(name="count")
            st.bar_chart(codepage_counts.set_index("codepage"))

            avg_conf = table["codepage_confidence"].mean()
            st.write(f"Average codepage confidence: {avg_conf:.3f}")

            # Span preview: show first record's spans with confidences if available
            if results:
                st.markdown("**Inspect record spans**")
                rec_idx = st.slider("Record index", 0, len(results) - 1, 0)
                first = results[rec_idx]
                span_rows = []
                for span in first.tag_spans:
                    conf = None
                    # map span start to confidence if we have them
                    if first.tag_confidences:
                        conf = round(
                            sum(first.tag_confidences[span["start"] : span["end"]]) / max(
                                1, (span["end"] - span["start"])
                            ),
                            3,
                        )
                    span_rows.append(
                        {"tag": span["tag"], "start": span["start"], "end": span["end"], "avg_confidence": conf}
                    )
                st.dataframe(pd.DataFrame(span_rows))
                # Tag distribution for this record
                tag_counts = pd.Series([s["tag"] for s in first.tag_spans]).value_counts()
                st.bar_chart(tag_counts)
                st.code(f"Boundary positions: {first.boundary_positions}")

            # Append to session logs
            st.session_state["infer_logs"].append(
                {
                    "bytes": len(data),
                    "records": len(results),
                    "seconds": elapsed,
                    "throughput_Bps": len(data) / max(elapsed, 1e-6),
                    "avg_cp_conf": avg_conf,
                }
            )
            if st.session_state["infer_logs"]:
                st.markdown("**Recent inference runs**")
                st.dataframe(pd.DataFrame(st.session_state["infer_logs"]).tail(5))

            if fmt == "json":
                buf = json.dumps(results, default=lambda o: o.__dict__).encode()
                st.download_button("Download JSON", buf, file_name="infer.json")
            elif fmt == "jsonl":
                buf = "\n".join(json.dumps(r.__dict__) for r in results).encode()
                st.download_button("Download JSONL", buf, file_name="infer.jsonl")
            else:
                arrow_path = Path("tmp.arrow")
                results_to_arrow(results, arrow_path)
                st.download_button("Download Arrow", arrow_path.read_bytes(), file_name="infer.arrow")

    with tabs[2]:
        st.subheader("Train (small demo)")
        st.write("Runs a tiny synthetic multi-head training job for demo purposes (CPU-friendly).")
        train_epochs = st.number_input("Epochs", min_value=1, value=1, step=1)
        samples_per_cp = st.number_input("Samples per codepage", min_value=4, value=8, step=1)
        device = st.selectbox("Device", ["cpu", "cuda"], key="train_device")
        output_dir = st.text_input("Output dir", "artifacts/ui-multihead")
        st.caption("For real training, switch to CLI with --real manifests to mix real datasets.")
        if st.button("Run tiny training (synthetic)"):
            cfg = MultiTaskConfig(
                output_dir=Path(output_dir),
                samples_per_codepage=int(samples_per_cp),
                epochs=int(train_epochs),
                device=device,
                max_len=96,
                batch_size=8,
            )
            start = time.time()
            metrics = train_multitask(cfg)
            elapsed = time.time() - start
            st.json(metrics)
            st.success(f"Training completed in {elapsed:.2f}s; checkpoint at {metrics['checkpoint']}")
            st.session_state["train_logs"].append(
                {
                    "epochs": train_epochs,
                    "samples_per_codepage": samples_per_cp,
                    "seconds": elapsed,
                    "checkpoint": metrics.get("checkpoint"),
                }
            )
        if st.session_state["train_logs"]:
            st.markdown("**Recent training runs**")
            st.dataframe(pd.DataFrame(st.session_state["train_logs"]).tail(5))


if __name__ == "__main__":
    main()
