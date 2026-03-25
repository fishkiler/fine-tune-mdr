"""
Data source fetchers for the MDR training pipeline.

Each fetcher pulls data from an external source and converts it to JSONL
format with {"messages": [{"role": "user", "content": "..."}, ...]} records.

Usage:
    python -m scripts.sources.fetch_cisa_kev --output data/sources/cisa_kev.jsonl
    python -m scripts.sources.fetch_mitre_stix --output data/sources/mitre_stix.jsonl
    python -m scripts.sources.fetch_attack_logs --with-sigma-hq

Then ingest:
    python scripts/ingest_data.py --jsonl data/sources/cisa_kev.jsonl --source cisa_kev
    python scripts/ingest_data.py --jsonl data/sources/attack_logs.jsonl --source attack_logs
"""
