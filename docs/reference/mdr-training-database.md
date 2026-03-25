# MDR Training Database Reference

## Overview

The MDR training database (`mdr-database/mdr_dataset.db`) is a SQLite database containing **365,452 curated training examples** for fine-tuning the Foundation-Sec-8B model into an MDR (Managed Detection and Response) security analyst. The data spans 10 security domains sourced from 8 different data pipelines.

**Schema Version:** 4
**Database Path:** `mdr-database/mdr_dataset.db`

## Domain Breakdown

| Domain | Records | Description |
|---|---:|---|
| `cve` | 350,072 | CVE vulnerability summaries, impact assessments, mitigations |
| `siem_queries` | 6,213 | Splunk SPL and KQL detection queries |
| `apt_intel` | 3,254 | APT/threat actor profiles, TTPs, attribution |
| `mitre_attack` | 2,694 | MITRE ATT&CK technique descriptions and detection guidance |
| `security_general` | 1,260 | General security best practices and guidance |
| `log_analysis` | 1,112 | Security log triage, correlation, and threat identification |
| `sigma_rules` | 499 | Sigma detection rule writing and explanation |
| `secure_code_review` | 348 | Secure code review examples |
| **Total** | **365,452** | |

## Data Sources

| Source | Records | What It Contains |
|---|---:|---|
| `cveorg` | 345,489 | CVE records from cve.org (2020-2025), quality-filtered to exclude URL-only mitigations |
| `spl_production` | 5,533 | Real production Splunk searches from a live MDR platform — 108 MDR detection rules, 2,826 scheduled searches, 371 AI-generated log analysis queries |
| `cisa_kev` | 4,578 | CISA Known Exploited Vulnerabilities catalog |
| `pentestds` | 2,901 | PentestDS dataset (MITRE, code review, security general) |
| `attack_logs` | 2,291 | Simulator-generated attack log analysis, SIEM queries, and Sigma rules from the MDR Log Simulator API |
| `mitre_stix_groups` | 1,667 | MITRE ATT&CK threat group profiles from STIX data |
| `misp_galaxy` | 1,611 | MISP Galaxy threat actor intelligence |
| `mitre_stix` | 1,382 | MITRE ATT&CK technique descriptions from STIX data |

## Database Schema

### Core Table: `all_records`

Every training example lives in this table with full metadata:

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER PK | Auto-increment ID |
| `split` | TEXT | `train`, `val`, or `test` |
| `domain` | TEXT | One of the 10 domains listed above |
| `question_type` | TEXT | Categorized question type (e.g., `summary`, `spl_query`, `log_triage`, `sigma_write`) |
| `user_message` | TEXT | The training question/prompt |
| `assistant_message` | TEXT | The training answer/response |
| `cve_ids` | TEXT | Extracted CVE IDs (comma-separated) |
| `mitre_techniques` | TEXT | Extracted MITRE technique IDs (comma-separated) |
| `cwe_ids` | TEXT | Extracted CWE IDs (comma-separated) |
| `severity` | TEXT | CRITICAL/HIGH/MEDIUM/LOW |
| `cvss_score` | REAL | CVSS base score (0.0-10.0) |
| `char_length` | INTEGER | Total character length of user+assistant |
| `content_hash` | TEXT UNIQUE | SHA-256 hash for deduplication |
| `validation_status` | TEXT | `pass`, `warn`, or `fail` from rule-based validation |
| `validation_errors` | TEXT | JSON array of validation issues |
| `quality_score` | REAL | LLM quality review score (1-5 scale) |
| `quality_scores` | TEXT | JSON with per-dimension scores |
| `quality_reviewed_at` | TEXT | Timestamp of LLM review |
| `quality_reviewer` | TEXT | Model used for review |
| `source` | TEXT | Which pipeline produced this record |
| `ingested_at` | TEXT | When the record was ingested |
| `exported_at` | TEXT | When the record was last exported for training |

### Domain Tables

Each domain has its own table (`cve`, `mitre_attack`, `siem_queries`, `log_analysis`, `sigma_rules`, etc.) with a subset of columns and a `master_id` foreign key back to `all_records`. These are populated during ingestion for domain-specific queries.

### Supporting Tables

- `domain_stats` — Aggregated statistics per domain (record count, avg quality, validation rates)
- `export_history` — Tracks every training data export (date, count, thresholds, weights used)
- `schema_info` — Schema version tracking for migrations

## Question Types

The `question_type` column categorizes what each training example teaches:

| Type | Count | Domain |
|---|---:|---|
| `other` | 262,476 | CVE (generic questions) |
| `summary` | 45,654 | CVE summaries |
| `impact` | 35,951 | CVE impact assessments |
| `mitigation` | 8,736 | CVE remediation guidance |
| `spl_query` | 5,976 | Splunk SPL query writing |
| `tactic_lookup` | 1,364 | MITRE tactic identification |
| `code_review` | 847 | Secure code review |
| `group_profile` | 806 | APT group profiles |
| `technique_description` | 643 | MITRE technique explanations |
| `log_triage` | 621 | Security log triage and analysis |
| `kql_query` | 462 | KQL query writing (Microsoft Sentinel) |
| `attribution` | 443 | Threat actor attribution |
| `ttp_mapping` | 375 | Technique-tactic-procedure mapping |
| `sigma_write` | 290 | Sigma rule writing |
| `sigma_explain` | 209 | Sigma rule explanation |
| `targeting` | 154 | APT targeting profiles |
| `software_analysis` | 132 | Malware/tool analysis |
| `alias_lookup` | 131 | Threat actor alias resolution |
| `log_correlation` | 93 | Multi-source log correlation |
| `detection_guidance` | 64 | Detection strategy guidance |
| `campaign_analysis` | 25 | APT campaign analysis |

## Detection Engineering Data (New in Schema v4)

Three domains were added specifically for detection engineering training:

### `log_analysis` (1,112 examples)

Teaches the model to analyze raw security logs and identify attacks.

- **Single-source triage:** Analyze a single log event (CrowdStrike, Defender, Palo Alto, etc.) and identify the MITRE technique, evidence, severity, and response actions
- **Multi-source correlation:** Correlate 2-6 events from different platforms to describe an attack chain
- **Benign/threat triage:** Review a batch of mixed benign and malicious events, identify the threats

Ground-truth fields (`mitre_technique`, `mitre_tactic`, `severity`, `threat_name`) are stripped from question-side logs so the model learns to analyze patterns, not read labels.

### `siem_queries` (6,213 examples)

Teaches the model to write and explain Splunk SPL and KQL detection queries.

- **SPL query writing:** Write detection queries with proper `index=`, `sourcetype=`, field filters
- **Correlation SPL:** Multi-source correlation searches joining across sourcetypes
- **KQL queries:** Microsoft Sentinel analytics rules
- **Production SPL (5,533):** Real scheduled searches from a live Splunk MDR platform including `streamstats`-based brute force detection, `iplocation` geo-anomaly rules, tstats accelerated data model queries, and 108 MDR-specific detection rules
- **Query explanation:** Break down existing SPL queries and explain their detection logic

### `sigma_rules` (499 examples)

Teaches the model to write and understand Sigma detection rules.

- **Sigma rule writing:** Generate complete YAML rules with title, logsource, detection, condition, ATT&CK tags
- **Sigma rule explanation:** Analyze existing SigmaHQ rules field-by-field (208 real rules from SigmaHQ repository covering 386 ATT&CK techniques)

## Data Quality

### Deduplication

Every record has a SHA-256 `content_hash` computed from normalized user+assistant text. A `UNIQUE INDEX` on this column prevents duplicate ingestion at the database level. Current state: 365,452 records, 365,452 unique hashes, zero duplicates.

### Rule-Based Validation

The `validation_status` column reflects automated format/content checks:

- **CVE:** Valid CVE ID format, CVSS range, severity-score consistency, CWE format, response substance
- **MITRE ATT&CK:** Valid technique ID format, tactic presence, response substance
- **Secure Code Review:** Code blocks present, security-specific language
- **APT Intel:** Group identity markers, specificity (technique IDs, tactics)
- **Log Analysis:** Embedded JSON logs in question, technique ID in answer, severity assessment
- **SIEM Queries:** Code block or query breakdown present, SPL/KQL syntax markers
- **Sigma Rules:** YAML structure markers (title/logsource/detection/condition)

### LLM Quality Review

A subset of records have been reviewed by Claude for quality scoring (1-5 scale across 5 dimensions). Average scores by domain:

| Domain | Avg Score | Records Reviewed |
|---|---|---|
| `secure_code_review` | 4.78 | 25 |
| `security_general` | 4.77 | 25 |
| `mitre_attack` | 4.40 | 25 |
| `cve` | 4.09 | 100 |

## Export Configuration

Training data export uses domain weights to balance the dataset:

```yaml
domain_weights:
  cve: 0.08          # ~28K from 350K (top quality only)
  mitre_attack: 10.0  # ~27K
  secure_code_review: 30.0  # ~10K
  security_general: 8.0     # ~10K
  apt_intel: 1.0
  exploitdb: 1.0
  stix_general: 1.0
  log_analysis: 5.0   # ~5.6K
  siem_queries: 8.0   # ~50K
  sigma_rules: 10.0   # ~5K
```

Quality thresholds filter out low-quality records:

```yaml
thresholds:
  default: 3.5
  cve: 3.0
  secure_code_review: 4.0
  siem_queries: 4.0
  sigma_rules: 4.0
```

## Pipeline Scripts

| Script | Purpose |
|---|---|
| `scripts/ingest_data.py` | Ingest JSONL/Arrow data into the database with dedup |
| `scripts/validate_data.py` | Run rule-based validation on all records |
| `scripts/review_data.py` | LLM quality scoring via Claude API |
| `scripts/export_training_data.py` | Quality-filtered export with domain weights |
| `scripts/db_utils.py` | Shared DB functions (hash, classify, schema, migrations) |
| `scripts/sources/fetch_attack_logs.py` | Generate log analysis/SIEM/Sigma data from MDR Log Simulator |
| `scripts/sources/sigma_hq.py` | Download and index SigmaHQ rules by ATT&CK technique |
| `scripts/sources/ingest_spl_db.py` | Convert production Splunk searches to training examples |
| `scripts/sources/ingest_cveorg.py` | Ingest CVE records from cve.org source files |
| `scripts/sources/fetch_mitre_stix.py` | Fetch MITRE ATT&CK STIX technique data |
| `scripts/sources/fetch_mitre_groups.py` | Fetch MITRE ATT&CK threat group data |
| `scripts/sources/fetch_misp_galaxy.py` | Fetch MISP Galaxy threat actor intelligence |
| `scripts/sources/fetch_cisa_kev.py` | Fetch CISA Known Exploited Vulnerabilities |

## Pipeline Flow

```
Sources → JSONL → ingest_data.py → validate_data.py → review_data.py → export_training_data.py → train
```

1. **Fetch/Generate:** Each source script produces a JSONL file in `data/sources/`
2. **Ingest:** `ingest_data.py --jsonl <file> --source <label>` loads into the database with dedup
3. **Validate:** `validate_data.py` runs domain-specific format checks
4. **Review:** `review_data.py` samples records for LLM quality scoring (optional)
5. **Export:** `export_training_data.py` applies quality thresholds and domain weights, writes training-ready data to `data/export/`

## Key Design Decisions

- **Ground-truth stripping:** Attack log training examples have MITRE technique/tactic/severity fields removed from question-side logs so the model learns analysis rather than label-reading
- **Content-hash dedup:** SHA-256 of normalized text prevents any duplicate training examples
- **Domain classification:** Records are auto-classified into domains based on content signals in both the question and answer text
- **Schema migrations:** The database supports incremental schema upgrades (v1→v2→v3→v4) so existing data is preserved when new domains are added
- **SigmaHQ integration:** 3,104 real Sigma detection rules downloaded from GitHub, indexed by ATT&CK technique ID (386 techniques), cached locally in `data/sources/sigma_cache/`
- **Production SPL queries:** Real-world Splunk searches from a live MDR platform provide authentic detection patterns that generated queries cannot replicate
