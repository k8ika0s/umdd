import json
from pathlib import Path

import pandas as pd
import streamlit as st

from umdd.inference import InferenceResult, infer_bytes, results_to_arrow, results_to_jsonl
from umdd.manifest import Manifest, load_manifest, validate_manifest


def _render_spans(res: InferenceResult) -> str:
    parts = []
    for span in res.tag_spans:
        parts.append(f"{span['tag']}[{span['start']}-{span['end']}]")
    return ", ".join(parts)


def main() -> None:
    st.title("UMDD Inference & Validation Playground")

    checkpoint = st.text_input("Checkpoint path", "artifacts/multihead/multihead.pt")
    max_records = st.number_input("Max records", min_value=1, value=2, step=1)
    include_conf = st.checkbox("Include confidences", value=True)
    fmt = st.selectbox("Output format", ["json", "jsonl", "arrow"])

    st.header("Validate manifest")
    manifest_file = st.file_uploader("Upload manifest (json/yaml)", type=["json", "yml", "yaml"])
    if manifest_file:
        tmp_path = Path("uploaded_manifest")
        tmp_path.write_bytes(manifest_file.read())
        mf = load_manifest(tmp_path)
        result = validate_manifest(mf)
        st.json(result)

    st.header("Inference")
    uploaded = st.file_uploader("Upload RDW/BDW file", type=["bin", "dat"])
    if uploaded:
        data = uploaded.read()
        st.info(f"Received {len(data)} bytes")
        results = infer_bytes(
            data,
            Path(checkpoint),
            max_records=int(max_records),
            include_confidence=include_conf,
        )
        # Display summary table
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

        # Download outputs
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


if __name__ == "__main__":
    main()
