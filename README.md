
# Universal Mainframe Data Decoder (UMDD)
## Full Technical Architecture Document

## Executive Abstract
Modernization pipelines consistently fail at their weakest link: raw mainframe data does not self-describe. Encoding varies by codepage, structure is hidden behind legacy formats, and mixed-mode binary/text datasets make traditional ASCII↔EBCDIC conversion brittle, lossy, and error-prone.

UMDD (Universal Mainframe Data Decoder) introduces an AI-driven, byte-level intelligence engine capable of detecting encoding, structure, record boundaries, numeric fields, packed-decimal formats, copybook-aligned layouts, and semantic meaning—without human intervention.

Running on IBM Z, LinuxONE, or distributed GPU systems, UMDD becomes a drop-in solution for high-confidence data extraction into UTF-8, Arrow, Parquet, JSON, SQL, or analytical pipelines.

UMDD transforms raw mainframe data into fully structured, validated modern data at scale.

## System Overview
UMDD architecture consists of:

- Byte Embedding Layer
- Codepage & Encoding Detector
- Field Type Classifier
- Boundary & Structure Inference Engine
- Contextual EBCDIC→UTF-8 Translator
- Structured Output Generator (Arrow, Parquet, JSON, SQL)

### High-Level Pipeline
RAW DATA → Byte Embeddings → Multi-Head Analysis → Translation → Structured Output

## Core Architecture

### Byte Embedding Layer
- 256-token vocabulary (one per byte)
- Embedding dimensions: 32–128
- Optimized for s390x vector instructions
- Optional positional encoding for RDW, BDW, copybook alignment

### Multi-Head Model Components

#### Codepage Detection Head
Predicts EBCDIC codepage (CP037, CP1047, CP500, etc.) with probability scoring.

#### Field-Type Classifier
Tags bytes as:
- TEXT_EBCDIC
- NUMERIC_PACKED (COMP-3)
- NUMERIC_BINARY (COMP/BIN)
- ZONED_DECIMAL
- CONTROL_CODE
- HEADER_FIELD
- STRUCTURAL_METADATA

Prevents corruption during conversion.

#### Numeric Structure Detection
Detects:
- packed decimal (COMP-3)
- binary int (COMP, COMP-5)
- zoned decimal (with signed overpunch)
- IBM date formats (CYYMMDD, YYDDD)

#### Record Boundary Detection
Uses:
- attention patterns
- sequential alignment
- heuristic + learned signals
- RL-enhanced synthetic training

Outputs byte ranges marking field boundaries.

### Structure Inference Engine
Identifies:
- field offsets and lengths
- repeating record groups
- nested structures (CICS, MQ, SMF, VSAM)
- PIC X / PIC 9 inference without copybooks

### Contextual Translation Head
Sequence-to-sequence EBCDIC→UTF-8 with:
- contextual disambiguation
- vocabulary priors
- anomaly correction
- robust handling of mixed-mode datasets

### Structured Output Generator
Can emit:
- JSON
- CSV
- SQL rowsets
- Apache Arrow tables
- Parquet files
- Iceberg records

## Training Pipeline

### Phase 1 — Byte Modeling
- masked byte prediction
- autoencoder reconstruction
- next-byte prediction

### Phase 2 — Mixed-Mode Labeling
- supervised tagging for text vs binary vs packed
- boundary detection
- segmentation of synthetic COBOL datasets

### Phase 3 — Translation Training
- parallel EBCDIC/UTF-8 corpora
- sequence alignment loss
- contextual reconstruction

### Phase 4 — Structure Inference
- synthetic copybook generator
- VSAM/DB2/SMF real-world samples
- multi-loss training for segmentation + classification

### Loss Functions
- Cross-entropy (codepage)
- Token CE (field-tagging)
- Boundary regression
- Seq2Seq + CTC (translation)
- Multi-label field boundary loss

## Runtime & Platform Architecture

### IBM Z / LinuxONE Runtime
Optimized for:
- Telum AI accelerator
- vector extensions
- USS deployment
- z/OSMF integration

### Cloud / Distributed Runtime
- Kubernetes
- Red Hat OpenShift Z
- GPU clusters
- watsonx.data integration

### Performance Targets
- Batch decode: 500–1200 MB/s
- Streaming latency: 2–5 ms
- Telum inference: <1 µs per 4KB block

### Memory Layout
- Arrow buffers for intermediate state
- zero-copy transforms
- vector-friendly memory alignment

## IBM Ecosystem Integration

### z/OS Integration
Supports:
- VSAM RLS
- SMF 30/70/110 flows
- DFSORT/ICETOOL pre/post-processing
- z/OS Connect EE pipelines

### CICS & MQ
Decodes:
- MQMD
- MQRFH2
- COMMAREA structures

### DB2
Handles:
- DSNTIAUL unload formats
- mixed-mode binary structures
- numeric compression

### watsonx.data
UMDD → Arrow/Parquet → watsonx.data ingest → analytics/AI

## Roadmap (MVP → GA)

### MVP
- Byte embeddings
- Codepage classifier
- Field classifier
- Basic translator
- JSON output

### Beta
- Packed decimal decoding
- Boundary detection
- Arrow output
- VSAM support

### GA
- Full structure inference
- Parquet output
- Telum acceleration
- watsonx.data connectors

## Risks & Mitigations
- Ambiguous codepages → confidence scoring
- Data corruption → anomaly head
- Complex COBOL → synthetic training
- Mixed-mode unforeseen formats → domain adapters

## Conclusion
UMDD transforms the modernization ecosystem by automating the hardest part: decoding legacy, unstructured, encoded mainframe data. It brings AI-driven understanding to byte streams, unlocking hybrid cloud, analytics, and AI workflows across IBM Z and LinuxONE.

