# Plan: Database-Pipeline Integration with AI Data Quality Review

## Context

The MDR dataset (621K records) currently lives in two disconnected places: Arrow files for training and an SQLite database for viewing. There's no deduplication, no data quality validation, and no way to incrementally add new data. The priority is **data quality** — ensuring the training data is accurate, well-formed, and useful before training a newer model.

## Pipeline Flow

```
pentestds / custom / new sources
        |
        v
  ingest_data.py  -->  SQLite DB  (INSERT OR IGNORE by content_hash)
        |
  validate_data.py  -->  Rule-based format checks (fast, all records)
        |
  review_data.py  -->  LLM quality scoring (sample or full pass)
        |
  export_training_data.py  -->  Arrow (only quality_score >= threshold)
        |
  train_native.py  (unchanged)
```

---

## Part 1: Data Quality System (PRIORITY)

### 1A. Format Validation (Rule-Based, Fast)

New script: `scripts/validate_data.py`

Automated checks run against every record in the database:

**CVE Domain Checks:**
- CVE ID is valid format (CVE-YYYY-NNNNN) and exists in known CVE list
- Assistant response actually contains the CVE summary (not generic text)
- CVSS score is within valid range (0.0-10.0)
- Severity label matches CVSS score (e.g., CRITICAL = 9.0-10.0)
- CWE ID references exist and are valid
- Response addresses the question type (impact question gets impact answer, not mitigation)

**MITRE ATT&CK Domain Checks:**
- Technique ID is valid format (T####.###) and exists in ATT&CK framework
- Tactic name is one of the 14 valid tactics
- Technique-to-tactic mapping is correct (T1071.001 → Command and Control)
- Description matches the actual ATT&CK technique description

**Secure Code Review Checks:**
- Response contains both vulnerable and secure code examples
- Code blocks are present and syntactically valid
- Security explanation accompanies the code
- Language in code matches language in the question

**Security General Checks:**
- Response is substantive (not generic filler)
- Contains actionable security guidance
- No hallucinated tool names or non-existent standards

**Validation status stored in database:**
```sql
ALTER TABLE all_records ADD COLUMN validation_status TEXT;  -- 'pass', 'fail', 'warn'
ALTER TABLE all_records ADD COLUMN validation_errors TEXT;  -- JSON array of issues found
```

### 1B. LLM Quality Scoring

New script: `scripts/review_data.py`

Uses an LLM (Claude API or local model) to score records for quality:

**Scoring dimensions (1-5 each):**
- **Accuracy**: Is the information factually correct?
- **Completeness**: Does the response fully address the question?
- **Clarity**: Is the response well-written and understandable?
- **Relevance**: Does the response stay on topic?
- **Usefulness**: Would this help a security analyst in practice?

**Overall quality_score** = average of all dimensions (1.0-5.0)

**Database columns:**
```sql
ALTER TABLE all_records ADD COLUMN quality_score REAL;       -- 1.0-5.0 composite
ALTER TABLE all_records ADD COLUMN quality_scores TEXT;       -- JSON: {"accuracy":4,"completeness":5,...}
ALTER TABLE all_records ADD COLUMN quality_reviewed_at TEXT;  -- timestamp
ALTER TABLE all_records ADD COLUMN quality_reviewer TEXT;     -- 'claude-sonnet-4-6', 'manual', etc.
```

**Implementation approach:**
- Run format validation first (fast, free, catches obvious issues)
- Then LLM review on records that pass format validation
- Use batching to minimize API costs (send 10-20 records per API call with structured output)
- Start with a sample (e.g., 1000 random records per domain) to assess baseline quality
- Expand to full dataset if needed

**Export threshold**: Only export records where `validation_status = 'pass'` AND `quality_score >= 3.5` (configurable in config.yaml).

### 1C. Quality Dashboard in Viewer

Add a quality tab to the HTML viewer showing:
- Distribution of quality scores by domain (histogram)
- Count of validation failures by error type
- Lowest-quality records for manual review
- Domain-level quality averages

---

## Part 2: Deduplication

Add `content_hash TEXT NOT NULL UNIQUE` to `all_records`:
- SHA-256 of normalized (lowercase, collapsed whitespace) user+assistant text
- `INSERT OR IGNORE` at ingestion time silently skips duplicates
- No separate dedup pass needed

---

## Part 3: Domain Balancing

Current distribution is heavily skewed:

| Domain | Records | % |
|--------|---------|---|
| CVE | 618,575 | 99.5% |
| MITRE ATT&CK | 1,317 | 0.2% |
| Security General | 1,236 | 0.2% |
| Secure Code Review | 348 | 0.1% |

**Three strategies (configurable in config.yaml):**

1. **Upsampling minority domains**: Repeat MITRE/code review records N times in the training export. Simple but risks overfitting on the small domain.

2. **Capping majority domain**: Limit CVE records to e.g., 50K highest-quality records. Reduces training time dramatically while keeping quality high. Combined with upsampling minorities.

3. **Curriculum learning**: Train in stages — first on diverse security data (balanced), then on the full CVE dataset. This gives the model a broad security foundation before specializing.

**Recommended approach**: Strategy 2 (cap + upsample). Configure in export:
```yaml
export:
  training_data:
    domain_weights:
      cve: 0.08          # Keep ~50K of 618K (top quality)
      mitre_attack: 10.0  # 1,317 × 10 = ~13K
      secure_code_review: 30.0  # 348 × 30 = ~10K
      security_general: 8.0     # 1,236 × 8 = ~10K
    quality_threshold: 3.5      # Only export records scoring >= 3.5
```

This produces a ~83K record training set that's roughly 60% CVE, 16% MITRE, 12% code review, 12% security general — much more balanced.

---

## Part 4: Adding New Data Sources

Beyond pentestds, the pipeline can ingest from any source that produces JSONL with `{"messages": [...]}` format.

**Potential new sources:**

| Source | Data Type | Access Method |
|--------|-----------|---------------|
| CISA KEV | Known exploited vulnerabilities | REST API (free, no key) |
| NIST NVD 2.0 API | CVE details + CVSS | REST API (free, key recommended) |
| MITRE ATT&CK STIX | Techniques, tactics, groups | GitHub JSON files |
| Rapid7 Vulnerability DB | Vuln summaries | REST API (may need key) |
| ExploitDB | Exploit code + descriptions | Git mirror / CSV |
| OWASP Testing Guide | Security testing guidance | Markdown files |
| CWE Database | Weakness descriptions | XML download |
| Sigma Rules | Detection rules | GitHub YAML files |

**How to add a new source:**
1. Write a fetcher function that pulls data and converts to `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}` format
2. Save as JSONL file
3. Run `python scripts/ingest_data.py --jsonl path/to/new_source.jsonl`
4. The ingestion script handles classification, metadata extraction, dedup, and insertion automatically

**Each new source gets a dedicated fetcher script** in `scripts/sources/`:
```
scripts/sources/
  fetch_cisa_kev.py
  fetch_nvd.py
  fetch_mitre_stix.py
  fetch_sigma_rules.py
```

---

## Part 5: Training Strategy

### Incremental Training (Fine-tune on New Data)

After adding new data to the database, you don't need to retrain from scratch:

1. **Export only new records**: `export_training_data.py --since 2026-02-21` exports records added after a date
2. **Continue from checkpoint**: `train_native.py` already supports `--resume-from-checkpoint`
3. **Short retraining**: Run 1-2 epochs on just the new data, starting from the existing adapter weights

### Full Retraining Triggers

Retrain from scratch when:
- Domain distribution changes significantly (e.g., added 10K+ MITRE records)
- Quality review flagged >5% of training data as low-quality and you removed it
- Base model is updated (new Foundation-Sec version)

### Training with Quality Scores

The database enables quality-aware training:
- **High-quality emphasis**: Export only records with `quality_score >= 4.0` for the first epoch, then include `>= 3.5` for the second epoch
- **Domain-specific quality thresholds**: Require higher quality for code review (4.0) vs CVE summaries (3.0) since code examples need to be correct
- **Rejection sampling**: If a record scored low on "accuracy" but high on everything else, it might have factual errors — exclude it

---

## Implementation Steps

### Phase 1: Database Schema + Dedup (Foundation)
1. Create `scripts/db_utils.py` — shared functions (hash, classify, schema)
2. Update `build_dataset_db.py` — new schema with content_hash, quality columns
3. Migrate existing 621K records into new schema

### Phase 2: Data Quality (PRIORITY)
4. Create `scripts/validate_data.py` — rule-based format validation
5. Create `scripts/review_data.py` — LLM quality scoring
6. Run validation on full database, review sample
7. Add quality dashboard to viewer

### Phase 3: Pipeline Integration
8. Create `scripts/ingest_data.py` — incremental ingestion with dedup
9. Create `scripts/export_training_data.py` — quality-filtered export with domain weighting
10. Modify `scripts/refresh_data.py` — rewire to use DB pipeline
11. Modify `scripts/clean_data.py` — operate on DB instead of Arrow

### Phase 4: New Sources + Training
12. Create source fetcher framework (`scripts/sources/`)
13. Add CISA KEV and MITRE STIX fetchers as first new sources
14. Test incremental training from new data

---

## Critical Files

| File | Action | Purpose |
|------|--------|---------|
| `scripts/db_utils.py` | CREATE | Shared DB functions |
| `scripts/validate_data.py` | CREATE | Rule-based quality checks |
| `scripts/review_data.py` | CREATE | LLM quality scoring |
| `scripts/ingest_data.py` | CREATE | Ingestion with dedup |
| `scripts/export_training_data.py` | CREATE | Quality-filtered export |
| `mdr-database/build_dataset_db.py` | MODIFY | New schema + migration |
| `mdr-database/view_dataset.py` | MODIFY | Quality dashboard |
| `scripts/refresh_data.py` | MODIFY | Rewire to DB pipeline |
| `scripts/clean_data.py` | MODIFY | Operate on DB |
| `config.yaml` | MODIFY | Add database/quality/export sections |

## Verification

1. **Schema migration**: 621K records populated with content_hash, 0 duplicates
2. **Validation run**: Format checks flag malformed records, counts reported per domain
3. **Quality review**: LLM scores a sample of 1000 records, scores stored in DB
4. **Dedup test**: Insert duplicate → silently skipped
5. **Export test**: Only clean, quality-passing records exported to Arrow
6. **Training test**: `train_native.py` loads exported Arrow files successfully
