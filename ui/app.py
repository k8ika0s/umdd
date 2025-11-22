import json
from pathlib import Path

import pandas as pd
import streamlit as st

from umdd.inference import infer_bytes, results_to_arrow, results_to_jsonl


def main() -> None:
    st.title("UMDD Inference/Validation Playground")

    checkpoint = st.text_input("Checkpoint path", "artifacts/multihead/multihead.pt")
    max_records = st.number_input("Max records", min_value=1, value=2, step=1)
    include_conf = st.checkbox("Include confidences", value=True)
    fmt = st.selectbox("Output format", ["json", "jsonl", "arrow"])

    uploaded = st.file_uploader("Upload RDW/BDW file", type=["bin", "dat"])
    if uploaded:
        data = uploaded.read()
        st.info(f"Received {len(data)} bytes")
        results = infer_bytes(data, Path(checkpoint), max_records=int(max_records), include_confidence=include_conf)
        # Display summary table
        table = pd.DataFrame(
            [
                {
                    "record_index": r.record_index,
                    "byte_length": r.byte_length,
                    "codepage": r.codepage,
                    "codepage_confidence": r.codepage_confidence,
                    "tag_spans": json.dumps(r.tag_spans),
                    "boundary_positions": r.boundary_positions,
                }
                for r in results
            ]
        )
        st.dataframe(table)
        # Download outputs
        if fmt == "json":
            st.download_button("Download JSON", json.dumps(results, default=lambda o: o.__dict__), file_name="infer.json")
        elif fmt == "jsonl":
            buf = "\n".join(json.dumps(r.__dict__) for r in results).encode()
            st.download_button("Download JSONL", buf, file_name="infer.jsonl")
        else:
            arrow_path = Path("tmp.arrow")
            results_to_arrow(results, arrow_path)
            st.download_button("Download Arrow", arrow_path.read_bytes(), file_name="infer.arrow")


if __name__ == "__main__":
    main()
