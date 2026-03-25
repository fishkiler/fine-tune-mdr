# K-LLM Cybersecurity Threat Detection Architecture

## Project Briefing Document

This document describes the full architecture and implementation plan for a K-LLM (multi-model) cybersecurity threat detection system. Use this as context to assist with building, training, and deploying the system.

---

## Background

This project builds on Cisco's **Foundation-Sec-8B**, a cybersecurity-specific language model. The goal is to create an advanced threat detection capability that uses multiple specialist models working together — inspired by Palantir CTO Shyam Sankar's K-LLM philosophy — rather than relying on a single general-purpose model.

The core idea: send each query to K domain-specific models simultaneously, aggregate their responses through a synthesis layer, and surface a best answer along with dissenting views. In cybersecurity, dissent is signal — the one model that disagrees may catch a real threat the others missed.

All deployment is **on-premises on a DGX Spark system with 128GB unified memory**. No cloud dependencies.

---

## Current State

A **generalist cybersecurity LoRA adapter** is currently being trained via QLoRA on the Foundation-Sec-8B base model. Training details:

- **Dataset**: `jason-oneal/mitre-stix-cve-exploitdb-dataset-alpaca-chatml-harmony` (245K examples)
- **Sources in dataset**: MITRE STIX, CVE, and ExploitDB combined
- **Method**: QLoRA fine-tuning using Unsloth Docker container
- **Hardware**: DGX Spark, 128GB unified memory
- **Format**: ChatML/Alpaca template

This generalist model will serve as the **synthesis layer** in the K-LLM architecture because it has seen all data sources blended together, giving it the cross-domain reasoning needed to evaluate and reconcile outputs from the specialist models.

---

## K-LLM Architecture Overview

### The Concept

Instead of one model handling everything, the system uses multiple specialist models (the "K" in K-LLM), each trained on a specific cybersecurity data source. A router directs queries to all relevant specialists, and a synthesis layer aggregates their responses.

### System Flow

```
Query → Router → ┬→ NIST NVD Specialist    ─┐
                  ├→ MITRE ATT&CK Specialist ─┤
                  ├→ ExploitDB Specialist     ─┤→ Synthesis Layer → Structured Output
                  ├→ APT Intel Specialist     ─┤
                  └→ CVE.org Specialist       ─┘
```

### Structured Output from Synthesis

The synthesis layer produces tiered results:

- **Consensus** (all models agree) → High confidence finding
- **Majority** (most agree, some dissent) → Medium confidence, worth investigating
- **Minority Alert** (only one model flagged it) → Could be noise OR the critical catch
- **Conflict Resolution** — Where models directly contradict each other, with reasoning from each
- **Source Attribution** — Which specialist contributed what to the final answer

---

## Specialist Models — LoRA Adapters

Each specialist is a **LoRA adapter** (~50-200MB) that plugs into the single Foundation-Sec-8B base model (~5GB quantized Q4). The base model is loaded once; adapters are hot-swapped in milliseconds.

### 1. NIST NVD Specialist

- **Purpose**: Vulnerability severity assessment and triage
- **Trained on**: NIST National Vulnerability Database entries
- **Expertise**: CVSS scoring, CPE product mapping, CWE classifications, severity analysis
- **Data source**: NVD 2.0 API (`https://services.nvd.nist.gov/rest/json/cves/2.0`)
- **Key capability**: "How severe is this vulnerability? What products are affected?"

### 2. MITRE ATT&CK Specialist

- **Purpose**: Tactics, techniques, and procedures (TTP) mapping
- **Trained on**: MITRE ATT&CK framework data in STIX 2.1 format
- **Expertise**: Attack pattern identification, kill chain stage mapping, technique relationships
- **Data sources**:
  - MITRE ATT&CK STIX data: `github.com/mitre-attack/attack-stix-data`
  - HuggingFace: `tumeteor/Security-TTP-Mapping` (600+ hierarchical classes)
  - HuggingFace: `sarahwei/cyber_MITRE_CTI_dataset_v15`
  - HuggingFace: `HoangCuongNguyen/CTI-to-MITRE-dataset`
  - HuggingFace: `cobo512/Mitre-ATTACK-reasoning-dataset`
- **Key capability**: "What attack pattern does this match? What stage of the kill chain?"

### 3. ExploitDB Specialist

- **Purpose**: Exploit analysis and weaponization assessment
- **Trained on**: ExploitDB proof-of-concept exploits and descriptions
- **Expertise**: Exploit code analysis, weaponization patterns, PoC assessment
- **Data source**: ExploitDB Git repository (`gitlab.com/exploit-database/exploitdb`)
- **Key capability**: "Is there a known exploit? How weaponizable is this vulnerability?"

### 4. APT Intelligence Specialist

- **Purpose**: Threat actor profiling and attribution
- **Trained on**: Multiple APT intelligence sources
- **Expertise**: APT group identification, campaign attribution, threat actor TTPs
- **Data sources**:
  - APT-ClaritySet (305 APT groups, 25,923 malware samples, 2006-2025)
  - AlienVault OTX (now LevelBlue) — community threat intelligence via REST API
  - MISP (Malware Information Sharing Platform) — APT galaxy clusters
  - Abuse.ch — MalwareBazaar, URLhaus, SSL Blacklist
  - CIRCL — phishing campaigns, malware analysis
  - Shadowserver Foundation — malware, IP, SSL cert data
  - PickupSTIX — STIX-formatted open-source intelligence feed
  - GitHub: `NewBee119/Attack-Technique-Dataset` (APT articles + ATT&CK descriptions)
- **Key capability**: "Which threat groups use this technique? What's the attribution confidence?"

### 5. CVE.org Specialist

- **Purpose**: Vulnerability identification and raw CVE record analysis
- **Trained on**: CVE.org records via CVE Services API
- **Expertise**: CVE identification, vulnerability descriptions, initial triage
- **Data source**: CVE Services API (date-range queries for new records)
- **Key capability**: "What is this vulnerability? What's its initial classification?"

### Synthesis Layer (Generalist Model)

- **Purpose**: Orchestrate, evaluate, and reconcile outputs from all specialists
- **Model**: The currently-training generalist LoRA (245K examples across all sources)
- **Why it's the synthesizer**: It has seen MITRE + CVE + ExploitDB data blended together, so it understands cross-domain relationships (e.g., how a CVE maps to an ATT&CK technique and relates to a known exploit)

---

## Memory Budget on DGX Spark

| Component | Memory | Notes |
|---|---|---|
| Foundation-Sec-8B base (Q4) | ~5 GB | Loaded once, shared across all adapters |
| 4-5x LoRA Adapters | ~400-500 MB | ~100MB each, hot-swappable in milliseconds |
| Synthesis Model (generalist LoRA) | ~5 GB | Currently training — the generalist adapter |
| Router + Vector DB (FAISS/ChromaDB) | ~2-4 GB | For RAG context retrieval |
| **Total in use** | **~15 GB** | **113 GB headroom for batching + context** |

---

## Automated Monthly Update Pipelines

Each data source has its own ingestion pipeline that runs monthly to keep the specialist adapters current with evolving threat intelligence.

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────┐
│              PER-SOURCE INGESTION                    │
│                                                     │
│  NIST NVD    → NVD 2.0 API (lastModStartDate)      │
│  MITRE ATT&CK → GitHub STIX repo (git diff)        │
│  ExploitDB   → Git repo (new files since last pull) │
│  CVE.org     → CVE Services API (date-range query)  │
│  APT Intel   → OTX API + MISP TAXII + Abuse.ch     │
│                                                     │
│              ↓                                      │
│                                                     │
│  SHARED PROCESSING LAYER                            │
│  1. Deduplicate against existing training data      │
│  2. Format to ChatML/Alpaca template                │
│  3. Quality filter (drop incomplete/garbage)        │
│  4. Generate instruction-response pairs             │
│  5. Validate against schema                         │
│  6. Version tag + timestamp                         │
│                                                     │
│              ↓                                      │
│                                                     │
│  TRAINING ORCHESTRATOR                              │
│  • Stop Ollama/vLLM/LM Studio (free memory)        │
│  • Memory pre-flight check                          │
│  • Full retrain of LoRA adapter (not incremental)   │
│  • Benchmark against holdout test set               │
│  • If pass → promote to production                  │
│  • If fail → rollback to previous adapter, alert    │
│  • Restart model serving services                   │
└─────────────────────────────────────────────────────┘
```

### Why Full Retrain (Not Incremental)

Incremental continual fine-tuning (training only on new data) risks catastrophic forgetting — the model drifts away from older knowledge. In cybersecurity, forgetting an older CVE or technique is unacceptable. Full retrain appends new data to the accumulated dataset and retrains the LoRA from scratch each month. On DGX Spark, this takes ~4-6 hours per adapter overnight.

### Monthly Schedule

```
Night 1 — Pipelines pull fresh data from all sources
Night 2 — Process, format, deduplicate, merge with existing datasets
Night 3 — Train NVD + CVE specialist adapters
Night 4 — Train ATT&CK + ExploitDB specialist adapters
Night 5 — Train APT-Intel adapter
Night 6 — Benchmark all adapters against holdout test set
Night 7 — If all pass → promote to production, restart services
```

The system serves with old adapters until new ones are validated. Zero downtime.

---

## Data Source API Details

### NIST NVD

- **API**: `https://services.nvd.nist.gov/rest/json/cves/2.0`
- **Incremental pull**: Use `lastModStartDate` and `lastModEndDate` parameters
- **Rate limit**: 5 requests per 30 seconds without API key, 50 with key
- **Format**: JSON with CVSS scores, CPE strings, CWE classifications
- **Note**: Legacy feed files being removed after August 2025; use 2.0 API

### MITRE ATT&CK

- **Source**: `github.com/mitre-attack/attack-stix-data`
- **Format**: STIX 2.0 and 2.1 JSON bundles
- **Incremental pull**: Git diff against last pull to catch new/updated techniques, groups, software
- **Python tools**: `mitreattack-python` library for querying
- **Coverage**: 700+ techniques, 140+ threat groups, enterprise/mobile/ICS matrices

### ExploitDB

- **Source**: `gitlab.com/exploit-database/exploitdb` (Git repo)
- **Incremental pull**: Git pull + diff for new exploit files
- **Coverage**: 50K+ exploits with PoC code
- **Format**: Individual files with metadata headers

### CVE.org

- **API**: CVE Services REST API
- **Incremental pull**: Query by date range for newly published records
- **Format**: CVE 5.x JSON schema

### APT Intelligence Sources

- **AlienVault OTX**: REST API at `otx.alienvault.com`, STIX/TAXII server available
- **MISP**: STIX/TAXII feeds with timestamp-based polling
- **Abuse.ch**: Daily updated feeds (MalwareBazaar, URLhaus, SSL Blacklist)
- **CIRCL**: Public threat advisories and malware analysis
- **Shadowserver Foundation**: Daily reports via API
- **PickupSTIX**: ~100 new STIX-formatted intelligence items per day

---

## Implementation Roadmap

### Phase 1: Foundation (IN PROGRESS)

- Complete generalist model training on 245K combined dataset
- Evaluate and benchmark on cybersecurity tasks
- Establish baseline performance metrics

### Phase 2: Specialist Training

- Split the 245K dataset by source (NIST examples, ATT&CK examples, ExploitDB examples)
- Enrich each split with additional source-specific datasets listed above
- Train individual LoRA adapters per domain using QLoRA on DGX Spark
- Benchmark each specialist against source-specific test sets

### Phase 3: K-LLM Assembly

- Build the query router (classifies incoming queries by type/domain)
- Implement the synthesis pipeline (aggregation, comparison, tiered output)
- Wire specialist adapters into the multi-model inference flow
- Test end-to-end with sample threat scenarios

### Phase 4: Pipeline Automation

- Build per-source data ingestion scripts (one per API)
- Create the shared processing/formatting layer
- Implement the training orchestrator with memory management
- Add benchmark validation and automatic promotion/rollback logic
- Schedule as cron jobs for monthly execution

### Phase 5: Production Deployment

- Full K-LLM system operational on DGX Spark
- Continuous threat intelligence via automated pipelines
- Zero-downtime adapter updates
- Monitoring and alerting for pipeline failures

---

## Key Technical Decisions

1. **LoRA over full fine-tuning**: Memory efficient (~100MB per adapter vs ~16GB per model copy), enables hot-swapping, trains in hours not days
2. **Full retrain over incremental**: Prevents catastrophic forgetting of older threat intelligence
3. **Foundation-Sec-8B as base**: Domain-specific pre-training gives a massive head start over generic models
4. **Generalist as synthesizer**: The currently-training model sees all domains, making it the natural choice for cross-domain reconciliation
5. **On-premises only**: Data sovereignty and security — no threat intelligence leaves the system

---

## Training Infrastructure Notes

- **Platform**: DGX Spark, 128GB unified memory
- **Container**: Unsloth Docker (official)
- **Method**: QLoRA (4-bit quantization + Low-Rank Adaptation)
- **Critical**: Stop all other model services (Ollama, vLLM, LM Studio) before training to prevent OOM crashes
- **Memory pre-flight**: Always verify available memory before launching training runs

---

## What To Build Next

If picking up this project, the priority order is:

1. **Evaluate the generalist model** once current training completes
2. **Split the 245K dataset by source** into separate training sets
3. **Download and format additional datasets** from HuggingFace (TTP-Mapping, CTI-to-MITRE, etc.)
4. **Train the first specialist LoRA adapter** (recommend starting with MITRE ATT&CK — richest data available)
5. **Build the ingestion script for NIST NVD** (simplest API, good proof of concept for the pipeline)
6. **Prototype the router and synthesis layer** (even a simple keyword-based router works as v1)
