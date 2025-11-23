import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

from umdd.inference import InferenceResult, infer_bytes, results_to_arrow, results_to_jsonl
from umdd.manifest import load_manifest, validate_manifest
from umdd.training.multitask import MultiTaskConfig, train_multitask


def _render_spans(res: InferenceResult) -> str:
    parts = []
    for span in res.tag_spans:
        parts.append(f"{span['tag']}[{span['start']}-{span['end']}]")
    return ", ".join(parts)


def main() -> None:
    st.title("UMDD Inference, Validation, and Training")

    tabs = st.tabs(["Validate", "Infer", "Train"])

    with tabs[0]:
        st.subheader("Validate manifest")
        manifest_file = st.file_uploader("Upload manifest (json/yaml)", type=["json", "yml", "yaml"])
        if manifest_file:
            tmp_path = Path("uploaded_manifest")
            tmp_path.write_bytes(manifest_file.read())
            mf = load_manifest(tmp_path)
            result = validate_manifest(mf)
            st.json(result)

    with tabs[1]:
        st.subheader("Inference")
        checkpoint = st.text_input("Checkpoint path", "artifacts/multihead/multihead.pt")
        max_records = st.number_input("Max records", min_value=1, value=2, step=1, key="infer_max_records")
        include_conf = st.checkbox("Include confidences", value=True, key="infer_conf")
        fmt = st.selectbox("Output format", ["json", "jsonl", "arrow"], key="infer_fmt")
        log_perf = st.checkbox("Log performance (throughput/latency)", value=True, key="infer_perf")

        uploaded = st.file_uploader("Upload RDW/BDW file", type=["bin", "dat"], key="infer_file")
        if uploaded:
            data = uploaded.read()
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
        train_epochs = st.number_input("Epochs", min_value=1, value=1, step=1)
        samples_per_cp = st.number_input("Samples per codepage", min_value=4, value=8, step=1)
        device = st.selectbox("Device", ["cpu", "cuda"], key="train_device")
        output_dir = st.text_input("Output dir", "artifacts/ui-multihead")
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


if __name__ == "__main__":
    main()
