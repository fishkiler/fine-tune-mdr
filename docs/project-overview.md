# Fine-Tune MDR — Project Overview

> This document is written for another AI assistant to quickly understand what this project does, how it works, and what it is capable of.

## What This Project Is

Fine-Tune MDR is an end-to-end pipeline for fine-tuning a cybersecurity LLM to serve as the intelligence backbone of a Managed Detection and Response (MDR) system. It takes the [Foundation-Sec-8B-Instruct](https://huggingface.co/fdtn-ai/Foundation-Sec-8B-Instruct) base model (a Llama 3.1-based security model) and specializes it on a curated multi-domain cybersecurity training dataset.

The project covers **every stage** of the ML lifecycle:

1. **Data sourcing** — automated ingestion from 8+ cybersecurity data sources
2. **Data quality** — rule-based validation + LLM-scored quality review
3. **Knowledge graph** — STIX 2.1 relationship graph linking all threat intelligence
4. **Training** — LoRA fine-tuning with hyperparameter benchmarking
5. **Evaluation** — automated metrics (exact match, technique/tactic accuracy, F1)
6. **Inference** — production API server with calibrated confidence and tiered alerting
7. **Dashboards** — three web UIs for monitoring, exploration, and data browsing

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                              │
│  NVD CVEs · MITRE ATT&CK · CISA KEV · SigmaHQ Rules            │
│  ExploitDB · MISP Galaxy · MDR Log Simulator · PentestDS        │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Ingestion       │  scripts/ingest_data.py
                    │  (dedup, hash)   │  scripts/sources/*.py
                    └────────┬────────┘
                             │
              ┌──────────────▼──────────────┐
              │  SQLite Database             │  mdr-database/mdr_dataset.db
              │  Schema v5 — 380K records    │
              │  + STIX graph layer          │
              └──────┬──────────────┬───────┘
                     │              │
          ┌──────────▼──────┐  ┌───▼──────────────────┐
          │  Validate        │  │  STIX Relationship    │
          │  (rule-based)    │  │  Graph (182K nodes,   │
          │                  │  │  20K edges)           │
          └──────────┬──────┘  └───┬──────────────────┘
                     │              │
          ┌──────────▼──────┐  ┌───▼──────────────────┐
          │  Review          │  │  Training Pair        │
          │  (LLM scoring)   │  │  Generation           │
          └──────────┬──────┘  │  (14 categories)      │
                     │         └───┬──────────────────┘
                     │              │
              ┌──────▼──────────────▼───┐
              │  Export                   │  scripts/export_training_data.py
              │  (quality >= 3.5,        │
              │   domain-weighted)       │
              └──────────┬──────────────┘
                         │
              ┌──────────▼──────────────┐
              │  Fine-Tuning             │  train_native.py + train.sh
              │  LoRA on Foundation-Sec  │  (cgroup-guarded)
              └──────────┬──────────────┘
                         │
              ┌──────────▼──────────────┐
              │  Inference Server        │  inference.py
              │  Calibrated confidence   │
              │  + tiered alerting       │
              └─────────────────────────┘
```

## Hardware

The primary training hardware is an **NVIDIA DGX Spark with GB10 GPU** (Blackwell sm_121, 128GB unified CPU/GPU memory). Key hardware-specific constraints:

- **Memory-bandwidth bound** workload (not compute-bound)
- BF16 is faster than 4-bit quantization on this hardware
- Flash Attention unavailable — uses SDPA attention instead
- `torch.compile()` causes recompilation spikes — disabled
- NVRM driver deadlocks on OOM — training runs inside a systemd cgroup with `MemoryMax=110G` to get clean Python OOM errors instead of system freezes
- Optimal config: **batch=16, seq=1024** yielding 890 tok/s at 34.1 GB peak

The project also supports training on cloud GPUs (H100/A100) via Colab notebooks.

## Data Sources

| Source | Script | What It Provides |
|--------|--------|------------------|
| NVD / CVE.org | `scripts/sources/ingest_cveorg.py` | CVE vulnerability descriptions and metadata (~618K records) |
| MITRE ATT&CK | `scripts/sources/fetch_mitre_stix.py`, `fetch_stix_objects.py` | Attack techniques, tactics, groups, campaigns (STIX 2.1 format) |
| CISA KEV | `scripts/sources/fetch_cisa_kev.py` | Known Exploited Vulnerabilities catalog |
| SigmaHQ | `scripts/sources/sigma_hq.py` | Detection rules covering 388 MITRE techniques |
| ExploitDB | Via PentestDS | Exploit code and proof-of-concept data |
| MISP Galaxy | `scripts/sources/fetch_misp_galaxy.py` | Threat actor profiles and APT cluster data |
| MDR Log Simulator | `scripts/sources/fetch_attack_logs.py` | Synthetic attack log training data (2,291 examples) |
| PentestDS | Via `scripts/refresh_data.py` | Additional pentest-domain training data |

## Database (SQLite, Schema v5)

The database at `mdr-database/mdr_dataset.db` holds all training data and the STIX relationship graph. Key tables:

- **`records`** — Core training data (instruction/response pairs with quality scores)
- **`stix_objects`** — 182K STIX nodes (vulnerabilities, techniques, groups, malware, campaigns, etc.)
- **`stix_relationships`** — 20K typed edges (uses, mitigates, detects, subtechnique-of, exploits, attributed-to)
- **`stix_training_links`** — 375K links connecting training records to STIX objects they reference
- **`log_analysis`, `siem_queries`, `sigma_rules`** — Domain-specific tables for detection engineering content
- **`export_history`** — Tracks every training data export run

Shared utilities in `scripts/db_utils.py` handle hashing, classification, schema migration (v1 through v5), and all database operations.

## STIX Relationship Graph

The STIX 2.1 knowledge graph (`scripts/stix_graph.py`) loads the entire graph into memory for fast traversal. It contains:

- **182K nodes**: 178K vulnerabilities, 941 intrusion sets, 693 malware families, 691 attack patterns, 691 detection strategies, 336 sigma rules, 106 data components, 91 tools, 52 campaigns, 44 mitigations, 14 tactics
- **20K edges**: 17K uses, 1.4K mitigates, 1K detects, 475 subtechnique-of, 96 exploits, 25 attributed-to
- **14,742 cross-domain training pairs** generated by graph traversal across 14 categories

The graph is built by the master orchestrator `scripts/build_stix_graph.py` which runs: fetch → link → generate.

## Key Scripts

| Script | Purpose |
|--------|---------|
| `train_native.py` | LoRA fine-tuning with transformers + peft + trl |
| `train.sh` | Cgroup-guarded training wrapper (prevents OOM system freeze) |
| `inference.py` | FastAPI production inference server with calibrated confidence |
| `eval.py` | Evaluation: exact match, technique accuracy, tactic accuracy, F1 |
| `calibrate.py` | Post-training temperature calibration |
| `bench_throughput.py` | Batch/sequence length throughput benchmarking |
| `scripts/ingest_data.py` | Incremental data ingestion with dedup |
| `scripts/validate_data.py` | Rule-based format validation (CVE, MITRE, code, log, SIEM, Sigma) |
| `scripts/review_data.py` | LLM quality scoring via Claude API (5 dimensions, 1-5 scale) |
| `scripts/export_training_data.py` | Quality-filtered, domain-weighted export with `--new-only` support |
| `scripts/build_stix_graph.py` | Master orchestrator for STIX graph pipeline |
| `scripts/stix_graph.py` | In-memory STIX graph library with CLI query interface |
| `scripts/stix_graph_server.py` | FastAPI server for interactive graph visualization |
| `scripts/generate_stix_training_pairs.py` | Cross-domain training pair generation from graph traversal |
| `scripts/link_stix_training.py` | 5 strategies linking training records to STIX objects |
| `scripts/refresh_data.py` | Full data refresh pipeline (fetch all sources + rebuild) |
| `mdr-database/build_dataset_db.py` | Database builder with `--migrate` support |
| `mdr-database/view_dataset.py` | Web-based dataset browser |

## Dashboards

### 1. Training Dashboard (port 28000)

**File**: `dashboard/server.py` + `dashboard/static/index.html`

A real-time training monitoring dashboard built with FastAPI + SSE (Server-Sent Events). Features:

- **Live loss/learning-rate charts** — updates every training step via SSE streaming
- **Training process management** — start/stop training from the browser, view live logs
- **Dataset refresh** — trigger data re-ingestion from all sources, monitor progress with streaming logs
- **Benchmark dashboard** (`/bench`) — visualize batch/sequence throughput sweeps with per-trial results
- **State persistence** — reconnecting clients receive full metrics history for chart reconstruction

The training script sends metrics to the dashboard via HTTP POST to `/log`. The dashboard fans them out to all connected browsers.

### 2. STIX Graph Visualization (port 11969)

**File**: `scripts/stix_graph_server.py` + `dashboard/static/stix_graph.html`

An interactive force-directed graph explorer for the STIX 2.1 threat intelligence knowledge graph. Features:

- **Search** — find any STIX object by name with tiered scoring (exact > prefix > substring)
- **Subgraph exploration** — BFS expansion up to 3 levels deep with intelligent pruning at 200 nodes
- **Node details** — click any node to see full STIX object metadata
- **Detection highlighting** — toggle view to show detection coverage (sigma rules, detection strategies) with glow effects
- **CORS-enabled API** — allows the MDR Log Simulator frontend (on a separate machine) to query campaign data
- **vis.js rendering** — force-directed layout with node sizing by degree centrality

### 3. Dataset Viewer (port 8080)

**File**: `mdr-database/view_dataset.py`

A standalone web viewer for browsing the training dataset database. Features:

- **Browse all training records** with pagination
- **Filter by domain** (CVE, MITRE, code review, log analysis, SIEM, Sigma, etc.)
- **Quality score filtering** — view only records above a quality threshold
- **Validation status** — see which records passed rule-based validation
- **Export history** — view past training data export runs with counts and thresholds

## Training Configuration

All configuration lives in `config.yaml`. Key settings:

- **Model**: `fdtn-ai/Foundation-Sec-8B-Instruct` (Llama 3.1 architecture)
- **LoRA**: r=16, alpha=32, dropout=0.05, targeting all attention + MLP projections
- **Training**: 2 epochs, cosine LR schedule, lr=1e-4, warmup=100 steps, BF16
- **Dataset**: 90/5/5 train/val/test split, seed 42
- **Quality thresholds**: default 3.5, code review 4.0, SIEM/Sigma 4.0, CVE 3.0
- **Domain weights**: code review 30x, MITRE/Sigma 10x, SIEM/security 8x, CVE 0.08x (downsampled)

## Inference Server

The inference server (`inference.py`) provides:

- **POST /analyze** — submit text for cybersecurity threat analysis
- **Calibrated confidence** — temperature-scaled logit probabilities
- **Tiered alerting**: auto_alert (>90%), needs_verification (>70%), human_review (>40%), log_only (<40%)
- **Structured responses** — returns technique IDs, tactic classifications, and confidence scores

## Inter-System Communication

The project communicates with the **MDR Log Simulator** running on a separate machine (192.168.1.67:6971) via:

- **REST API** — fetching synthetic attack log training data
- **NFS mount** — `/mnt/ai_projects/mdr-campaigns/` is a shared directory for exchanging markdown files between AI sessions working on different parts of the system
- **CORS** — the STIX graph server allows cross-origin requests from the simulator's frontend

## Pipeline Commands

```bash
# Full data refresh (fetch all sources + rebuild database)
python scripts/refresh_data.py

# Build STIX graph (fetch → link → generate training pairs)
python -m scripts.build_stix_graph

# Validate and review data quality
python scripts/validate_data.py
python scripts/review_data.py --sample 100

# Export training data (quality-filtered)
python scripts/export_training_data.py
python scripts/export_training_data.py --new-only

# Train (with cgroup memory guard)
bash train.sh

# Start dashboards
uvicorn dashboard.server:app --host 0.0.0.0 --port 28000  # Training dashboard
python -m scripts.stix_graph_server --port 11969            # STIX graph
python mdr-database/view_dataset.py --port 8080             # Dataset viewer

# Evaluate
python eval.py

# Serve inference
python inference.py
```

## Key Design Decisions

- **SQLite over Postgres** — single-file database simplifies deployment on edge hardware; the entire dataset (380K records) fits comfortably
- **In-memory STIX graph** — loaded at server startup for sub-millisecond traversal; 182K nodes + 20K edges fit in ~500MB RAM
- **Domain weighting** — training data is resampled by domain importance (code review examples upsampled 30x, CVEs downsampled to ~8%) to balance the dataset despite CVEs dominating raw count
- **Cgroup memory guard** — wraps training in a systemd cgroup to prevent NVIDIA driver deadlocks on OOM, which would otherwise require a hard reboot
- **SSE over WebSockets** — simpler protocol for unidirectional metrics streaming; dashboard reconnects with full history replay
- **Quality gating** — every training record passes through rule-based validation AND LLM quality scoring before export; this prevents garbage-in/garbage-out
