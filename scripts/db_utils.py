"""
============================================================================
Fine-Tune MDR — Database Utilities
============================================================================
Shared functions for database operations: hashing, classification,
metadata extraction, schema management, and connection helpers.

Used by: build_dataset_db.py, ingest_data.py, validate_data.py,
         review_data.py, export_training_data.py
============================================================================
"""

import hashlib
import json
import logging
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_DIR = PROJECT_ROOT / "mdr-database"
DEFAULT_DB_PATH = DB_DIR / "mdr_dataset.db"

# ── Schema Version ──
SCHEMA_VERSION = 6  # v6 adds game adapter tables

# ── Domain List ──
DOMAINS = [
    "cve", "mitre_attack", "secure_code_review", "apt_intel",
    "exploitdb", "stix_general", "security_general",
    "log_analysis", "siem_queries", "sigma_rules",
]


# ============================================================================
# Content Hashing (for deduplication)
# ============================================================================

def content_hash(user_msg: str, assistant_msg: str) -> str:
    """Generate SHA-256 hash of normalized user+assistant text for dedup."""
    normalized = " ".join(f"{user_msg} {assistant_msg}".lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# ============================================================================
# Domain Classification
# ============================================================================

def classify_record(user_msg: str, assistant_msg: str) -> str:
    """Classify a record into a domain based on content."""
    full_text = f"{user_msg} {assistant_msg}"
    user_lower = user_msg.lower()

    # Detection engineering domains — checked first since these records
    # also contain technique IDs that would match mitre_attack otherwise.
    sigma_signals = ["sigma rule", "sigma detection", "write a sigma", "logsource:", "detection:"]
    siem_signals = ["write a splunk", "write a spl", "write a kql", "detection query",
                    "correlation search", "write a correlation", "detection rule for",
                    "detection engineer", "analytics rule", "spl query", "kql query",
                    "splunk spl", "spl correlation", "spl detection",
                    "sentinel kql", "sentinel analytics", "splunk search",
                    "splunk detection", "splunk administrator", "tstats query",
                    "tstats search", "data model", "this splunk spl",
                    "this splunk detection", "this splunk search"]
    # Also match "```spl" in assistant content for explain-type SIEM records
    asst_lower = assistant_msg.lower()
    siem_answer_signals = ["```spl", "```splunk", "sourcetype=", "index=main"]
    log_signals = ["analyze the following", "triage this", "identify any malicious",
                   "correlate them", "correlated security events", "analyze this",
                   "correlating alerts", "triaging a batch", "triage result",
                   "review the following", "correlate these", "security events from different",
                   "mdr analyst"]
    if any(sig in user_lower for sig in sigma_signals):
        return "sigma_rules"
    if any(sig in user_lower for sig in siem_signals):
        return "siem_queries"
    if any(sig in asst_lower for sig in siem_answer_signals):
        return "siem_queries"
    if any(sig in user_lower for sig in log_signals):
        return "log_analysis"

    # APT / Threat Actor intelligence — checked BEFORE CVE so that
    # APT records mentioning CVEs in context aren't misclassified.
    # Only checks the user message for group-related signals.
    apt_signals = [
        "threat actor", "threat group", "apt group", "brief me on",
        "who is", "which group", "nation-state", "intrusion set",
        "what malware", "what tools does", "what country",
        "threat intelligence", "state sponsor", "what sectors",
        "known victims", "known aliases", "same group as",
        "targeting profile", "software arsenal",
    ]
    if (re.search(r"APT\d+", user_msg) or
        re.search(r"\bG\d{4}\b", full_text) or
        "intrusion-set" in full_text or
        any(sig in user_lower for sig in apt_signals)):
        return "apt_intel"

    # CVE records (dominant domain)
    if re.search(r"CVE-\d{4}-\d+", full_text):
        return "cve"

    # MITRE ATT&CK (technique-focused)
    if re.search(r"T\d{4}(?:\.\d{3})?", full_text) or "ATT&CK" in full_text or "attack-pattern" in full_text:
        return "mitre_attack"

    # Secure code review (code snippets + security questions)
    if any(kw in user_msg for kw in ["Is this", "Is using", "Is parsing", "safe?", "secure?", "code review"]):
        return "secure_code_review"

    if "ExploitDB" in full_text or "exploit-db" in full_text.lower() or "EDB-ID" in full_text:
        return "exploitdb"

    if "STIX" in full_text or "TAXII" in full_text:
        return "stix_general"

    # General security best practices
    return "security_general"


# ============================================================================
# Metadata Extraction
# ============================================================================

def extract_cve_ids(text: str) -> str | None:
    """Extract all CVE IDs from text."""
    ids = sorted(set(re.findall(r"CVE-\d{4}-\d+", text)))
    return ",".join(ids) if ids else None


def extract_mitre_techniques(text: str) -> str | None:
    """Extract MITRE ATT&CK technique IDs from text."""
    ids = sorted(set(re.findall(r"T\d{4}(?:\.\d{3})?", text)))
    return ",".join(ids) if ids else None


def extract_cwe_ids(text: str) -> str | None:
    """Extract CWE IDs from text."""
    ids = sorted(set(re.findall(r"CWE-\d+", text)))
    return ",".join(ids) if ids else None


def extract_severity(text: str) -> str | None:
    """Extract CVSS severity from text."""
    m = re.search(r"Severity:\s*(CRITICAL|HIGH|MEDIUM|LOW)", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"Severity\s*:\s*(\w+)", text, re.IGNORECASE)
    if m and m.group(1).upper() in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
        return m.group(1).upper()
    return None


def extract_cvss_score(text: str) -> float | None:
    """Extract CVSS base score from text."""
    m = re.search(r"(?:CVSS\s+)?base\s+score:\s*([\d.]+)", text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


def extract_question_type(user_msg: str) -> str:
    """Classify the type of question being asked."""
    msg = user_msg.lower()

    # Detection engineering question types
    if "sigma" in msg and ("explain" in msg or "what does" in msg):
        return "sigma_explain"
    if "sigma" in msg and ("write" in msg or "create" in msg or "detect" in msg):
        return "sigma_write"
    if "kql" in msg or "kusto" in msg or "sentinel" in msg:
        return "kql_query"
    if "spl" in msg or "splunk" in msg or "correlation search" in msg:
        return "spl_query"
    if "correlate" in msg or "correlation" in msg:
        return "log_correlation"
    if any(p in msg for p in ["triage", "analyze the following", "analyze this",
                               "identify any malicious", "which are suspicious"]):
        return "log_triage"

    # APT / threat actor question types
    if any(p in msg for p in ["who is", "brief me on", "threat intelligence briefing",
                               "overview of", "what is known about the threat"]):
        return "group_profile"
    if "what techniques does" in msg or ("techniques" in msg and "use" in msg):
        return "ttp_mapping"
    if "detect" in msg and any(p in msg for p in ["activity", "monitor", "indicators"]):
        return "detection_guidance"
    if any(p in msg for p in ["what malware", "what tools", "software arsenal"]):
        return "software_analysis"
    if "campaign" in msg and any(p in msg for p in ["describe", "analysis", "what is known"]):
        return "campaign_analysis"
    if any(p in msg for p in ["what country", "state sponsor", "nation-state",
                               "operate from", "attribution"]):
        return "attribution"
    if any(p in msg for p in ["same group", "aliases", "also known", "tracked"]):
        return "alias_lookup"
    if any(p in msg for p in ["what sectors", "what industries", "victims",
                               "targeting profile"]):
        return "targeting"

    # CVE / general question types
    if "impact" in msg and "serious" in msg:
        return "impact"
    if "mitigat" in msg:
        return "mitigation"
    if "summarize" in msg or "summary" in msg:
        return "summary"
    if "explain" in msg and "plain language" in msg:
        return "explanation"
    if "tactic" in msg and "belong" in msg:
        return "tactic_lookup"
    if "describe the technique" in msg:
        return "technique_description"
    if "safe" in msg or "secure" in msg or "Is this" in msg:
        return "code_review"
    return "other"


def extract_all_metadata(user_msg: str, assistant_msg: str) -> dict:
    """Extract all metadata from a user/assistant message pair."""
    full_text = f"{user_msg} {assistant_msg}"
    return {
        "domain": classify_record(user_msg, assistant_msg),
        "question_type": extract_question_type(user_msg),
        "cve_ids": extract_cve_ids(full_text),
        "mitre_techniques": extract_mitre_techniques(full_text),
        "cwe_ids": extract_cwe_ids(full_text),
        "severity": extract_severity(assistant_msg),
        "cvss_score": extract_cvss_score(assistant_msg),
        "char_length": len(full_text),
        "content_hash": content_hash(user_msg, assistant_msg),
    }


# ============================================================================
# Schema Management
# ============================================================================

def _create_stix_tables(cur: sqlite3.Cursor) -> None:
    """Create STIX relationship graph tables (v5)."""
    cur.execute("""
        CREATE TABLE IF NOT EXISTS stix_objects (
            stix_id TEXT PRIMARY KEY,
            type TEXT NOT NULL,
            name TEXT NOT NULL,
            aliases TEXT,
            description TEXT,
            external_ids TEXT,
            source TEXT NOT NULL,
            platforms TEXT,
            kill_chain_phases TEXT,
            severity TEXT,
            cvss_score REAL,
            raw_stix_json TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS stix_relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            relationship_id TEXT UNIQUE,
            source_ref TEXT NOT NULL,
            target_ref TEXT NOT NULL,
            relationship_type TEXT NOT NULL,
            description TEXT,
            source TEXT NOT NULL,
            confidence INTEGER,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (source_ref) REFERENCES stix_objects(stix_id),
            FOREIGN KEY (target_ref) REFERENCES stix_objects(stix_id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS stix_training_links (
            stix_id TEXT NOT NULL,
            record_id INTEGER NOT NULL,
            link_type TEXT NOT NULL,
            confidence REAL DEFAULT 1.0,
            created_at TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (stix_id, record_id),
            FOREIGN KEY (stix_id) REFERENCES stix_objects(stix_id),
            FOREIGN KEY (record_id) REFERENCES all_records(id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS stix_stats (
            type TEXT PRIMARY KEY,
            object_count INTEGER DEFAULT 0,
            relationship_count INTEGER DEFAULT 0,
            training_link_count INTEGER DEFAULT 0,
            last_updated TEXT
        )
    """)
    # Performance indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_stix_obj_type ON stix_objects(type)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_stix_obj_name ON stix_objects(name)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_stix_obj_source ON stix_objects(source)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_stix_rel_source ON stix_relationships(source_ref)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_stix_rel_target ON stix_relationships(target_ref)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_stix_rel_type ON stix_relationships(relationship_type)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_stix_rel_source_type ON stix_relationships(source_ref, relationship_type)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_stix_rel_target_type ON stix_relationships(target_ref, relationship_type)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_stix_link_stix ON stix_training_links(stix_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_stix_link_record ON stix_training_links(record_id)")


def _create_game_tables(cur: sqlite3.Cursor) -> None:
    """Create game adapter tables (v6)."""
    cur.execute("""
        CREATE TABLE IF NOT EXISTS game_adapters (
            id INTEGER PRIMARY KEY,
            game_name TEXT NOT NULL UNIQUE,
            base_model TEXT NOT NULL,
            adapter_path TEXT,
            adapter_version INTEGER DEFAULT 0,
            frame_resolution TEXT,
            action_space TEXT NOT NULL,
            status TEXT DEFAULT 'collecting',
            total_frames INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            trained_at TIMESTAMP,
            eval_score_avg REAL,
            eval_score_best REAL,
            notes TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS game_frames (
            id INTEGER PRIMARY KEY,
            game_name TEXT NOT NULL,
            frame_path TEXT NOT NULL,
            frame_hash TEXT NOT NULL,
            action_label TEXT NOT NULL,
            action_id INTEGER NOT NULL,
            episode_id INTEGER,
            frame_index INTEGER,
            cumulative_score REAL,
            quality_score REAL,
            validated INTEGER DEFAULT 0,
            excluded INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (game_name) REFERENCES game_adapters(game_name)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS game_training_runs (
            id INTEGER PRIMARY KEY,
            game_name TEXT NOT NULL,
            adapter_version INTEGER NOT NULL,
            frames_used INTEGER,
            epochs INTEGER,
            lora_r INTEGER,
            lora_alpha INTEGER,
            learning_rate REAL,
            final_loss REAL,
            training_time_seconds REAL,
            eval_avg_score REAL,
            eval_best_score REAL,
            config_snapshot TEXT,
            started_at TIMESTAMP,
            completed_at TIMESTAMP,
            FOREIGN KEY (game_name) REFERENCES game_adapters(game_name)
        )
    """)
    # Performance indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_game_frames_game ON game_frames(game_name)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_game_frames_hash ON game_frames(frame_hash)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_game_frames_episode ON game_frames(game_name, episode_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_game_frames_action ON game_frames(game_name, action_label)")


def create_schema(conn: sqlite3.Connection) -> None:
    """Create the full database schema (v2 with quality + dedup columns)."""
    cur = conn.cursor()

    # Schema version tracking
    cur.execute("""
        CREATE TABLE IF NOT EXISTS schema_info (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    cur.execute(
        "INSERT OR REPLACE INTO schema_info (key, value) VALUES (?, ?)",
        ("version", str(SCHEMA_VERSION)),
    )
    cur.execute(
        "INSERT OR REPLACE INTO schema_info (key, value) VALUES (?, ?)",
        ("created_at", datetime.now(timezone.utc).isoformat()),
    )

    # Master table with all records
    cur.execute("""
        CREATE TABLE IF NOT EXISTS all_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            split TEXT NOT NULL,
            domain TEXT NOT NULL,
            question_type TEXT,
            user_message TEXT,
            assistant_message TEXT,
            cve_ids TEXT,
            mitre_techniques TEXT,
            cwe_ids TEXT,
            severity TEXT,
            cvss_score REAL,
            char_length INTEGER,
            content_hash TEXT NOT NULL,
            validation_status TEXT,
            validation_errors TEXT,
            quality_score REAL,
            quality_scores TEXT,
            quality_reviewed_at TEXT,
            quality_reviewer TEXT,
            source TEXT DEFAULT 'pentestds',
            ingested_at TEXT,
            exported_at TEXT
        )
    """)

    # Unique index on content_hash for deduplication
    cur.execute("""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_content_hash
        ON all_records(content_hash)
    """)

    # Domain-specific tables
    for domain in DOMAINS:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {domain} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                master_id INTEGER,
                split TEXT NOT NULL,
                question_type TEXT,
                user_message TEXT,
                assistant_message TEXT,
                cve_ids TEXT,
                mitre_techniques TEXT,
                cwe_ids TEXT,
                severity TEXT,
                cvss_score REAL,
                char_length INTEGER,
                FOREIGN KEY (master_id) REFERENCES all_records(id)
            )
        """)

    # Stats table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS domain_stats (
            domain TEXT PRIMARY KEY,
            record_count INTEGER,
            avg_length REAL,
            total_cves INTEGER,
            total_techniques INTEGER,
            severity_critical INTEGER DEFAULT 0,
            severity_high INTEGER DEFAULT 0,
            severity_medium INTEGER DEFAULT 0,
            severity_low INTEGER DEFAULT 0,
            avg_quality_score REAL,
            validated_count INTEGER DEFAULT 0,
            validation_pass_count INTEGER DEFAULT 0,
            validation_fail_count INTEGER DEFAULT 0
        )
    """)

    # Export history table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS export_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            exported_at TEXT NOT NULL,
            record_count INTEGER NOT NULL,
            quality_threshold REAL,
            domain_weights TEXT,
            since_filter TEXT,
            output_path TEXT,
            format TEXT,
            domain_counts TEXT,
            notes TEXT
        )
    """)

    # STIX relationship graph layer (v5)
    _create_stix_tables(cur)

    # Game adapter tables (v6)
    _create_game_tables(cur)

    conn.commit()


def create_indexes(conn: sqlite3.Connection) -> None:
    """Create indexes for faster queries."""
    cur = conn.cursor()

    cur.execute("CREATE INDEX IF NOT EXISTS idx_all_domain ON all_records(domain)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_all_split ON all_records(split)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_all_cve ON all_records(cve_ids)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_all_severity ON all_records(severity)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_all_qtype ON all_records(question_type)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_all_quality ON all_records(quality_score)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_all_validation ON all_records(validation_status)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_all_source ON all_records(source)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_all_ingested ON all_records(ingested_at)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_all_exported ON all_records(exported_at)")

    for domain in DOMAINS:
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{domain}_cve ON {domain}(cve_ids)")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{domain}_severity ON {domain}(severity)")
        cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{domain}_split ON {domain}(split)")

    conn.commit()


def migrate_schema(conn: sqlite3.Connection) -> None:
    """Migrate schema to current version (handles v1→v2→v3)."""
    cur = conn.cursor()

    # Determine current version
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='schema_info'")
    if cur.fetchone() is None:
        current_version = 1
    else:
        cur.execute("SELECT value FROM schema_info WHERE key = 'version'")
        row = cur.fetchone()
        current_version = int(row[0]) if row else 1

    if current_version >= SCHEMA_VERSION:
        log.info(f"Schema already at v{current_version}, no migration needed.")
        return

    # ── v1 → v2: content_hash, quality columns, source tracking ──
    if current_version < 2:
        log.info("Migrating schema from v1 to v2...")

        new_columns = [
            ("content_hash", "TEXT NOT NULL DEFAULT ''"),
            ("validation_status", "TEXT"),
            ("validation_errors", "TEXT"),
            ("quality_score", "REAL"),
            ("quality_scores", "TEXT"),
            ("quality_reviewed_at", "TEXT"),
            ("quality_reviewer", "TEXT"),
            ("source", "TEXT DEFAULT 'pentestds'"),
            ("ingested_at", "TEXT"),
        ]

        for col_name, col_def in new_columns:
            try:
                cur.execute(f"ALTER TABLE all_records ADD COLUMN {col_name} {col_def}")
                log.info(f"  Added column: {col_name}")
            except sqlite3.OperationalError:
                pass  # Column already exists

        stats_columns = [
            ("avg_quality_score", "REAL"),
            ("validated_count", "INTEGER DEFAULT 0"),
            ("validation_pass_count", "INTEGER DEFAULT 0"),
            ("validation_fail_count", "INTEGER DEFAULT 0"),
        ]
        for col_name, col_def in stats_columns:
            try:
                cur.execute(f"ALTER TABLE domain_stats ADD COLUMN {col_name} {col_def}")
            except sqlite3.OperationalError:
                pass

        cur.execute("""
            CREATE TABLE IF NOT EXISTS schema_info (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        conn.commit()
        log.info("  v1 → v2 migration complete.")

    # ── v2 → v3: exported_at column, export_history table ──
    if current_version < 3:
        log.info("Migrating schema from v2 to v3...")

        try:
            cur.execute("ALTER TABLE all_records ADD COLUMN exported_at TEXT")
            log.info("  Added column: exported_at")
        except sqlite3.OperationalError:
            pass  # Column already exists

        cur.execute("""
            CREATE TABLE IF NOT EXISTS export_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                exported_at TEXT NOT NULL,
                record_count INTEGER NOT NULL,
                quality_threshold REAL,
                domain_weights TEXT,
                since_filter TEXT,
                output_path TEXT,
                format TEXT,
                domain_counts TEXT,
                notes TEXT
            )
        """)
        log.info("  Created table: export_history")

        cur.execute("CREATE INDEX IF NOT EXISTS idx_all_exported ON all_records(exported_at)")

        # Backfill ingested_at for migrated records that have NULL
        cur.execute("SELECT COUNT(*) FROM all_records WHERE ingested_at IS NULL")
        null_count = cur.fetchone()[0]
        if null_count > 0:
            backfill_ts = datetime.now(timezone.utc).isoformat()
            cur.execute(
                "UPDATE all_records SET ingested_at = ? WHERE ingested_at IS NULL",
                (backfill_ts,),
            )
            log.info(f"  Backfilled ingested_at for {null_count:,} records")

        conn.commit()
        log.info("  v2 → v3 migration complete.")

    # ── v3 → v4: log_analysis, siem_queries, sigma_rules domain tables ──
    if current_version < 4:
        log.info("Migrating schema from v3 to v4...")
        new_domains = ["log_analysis", "siem_queries", "sigma_rules"]
        for domain in new_domains:
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {domain} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    master_id INTEGER,
                    split TEXT NOT NULL,
                    question_type TEXT,
                    user_message TEXT,
                    assistant_message TEXT,
                    cve_ids TEXT,
                    mitre_techniques TEXT,
                    cwe_ids TEXT,
                    severity TEXT,
                    cvss_score REAL,
                    char_length INTEGER,
                    FOREIGN KEY (master_id) REFERENCES all_records(id)
                )
            """)
            cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{domain}_cve ON {domain}(cve_ids)")
            cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{domain}_severity ON {domain}(severity)")
            cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{domain}_split ON {domain}(split)")
            log.info(f"  Created table: {domain}")
        conn.commit()
        log.info("  v3 → v4 migration complete.")

    # ── v4 → v5: STIX relationship graph layer ──
    if current_version < 5:
        log.info("Migrating schema from v4 to v5...")
        _create_stix_tables(cur)
        conn.commit()
        log.info("  v4 → v5 migration complete.")

    # ── v5 → v6: Game adapter tables ──
    if current_version < 6:
        log.info("Migrating schema from v5 to v6...")
        _create_game_tables(cur)
        conn.commit()
        log.info("  v5 → v6 migration complete.")

    # Update version
    cur.execute("""
        CREATE TABLE IF NOT EXISTS schema_info (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    cur.execute(
        "INSERT OR REPLACE INTO schema_info (key, value) VALUES (?, ?)",
        ("version", str(SCHEMA_VERSION)),
    )
    cur.execute(
        "INSERT OR REPLACE INTO schema_info (key, value) VALUES (?, ?)",
        ("migrated_at", datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()
    log.info(f"Schema version set to v{SCHEMA_VERSION}.")


def backfill_content_hashes(conn: sqlite3.Connection, batch_size: int = 10000) -> int:
    """Backfill content_hash for records that don't have one."""
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM all_records WHERE content_hash = '' OR content_hash IS NULL")
    total = cur.fetchone()[0]

    if total == 0:
        return 0

    log.info(f"Backfilling content hashes for {total:,} records...")
    updated = 0

    while True:
        cur.execute(
            "SELECT id, user_message, assistant_message FROM all_records "
            "WHERE content_hash = '' OR content_hash IS NULL LIMIT ?",
            (batch_size,),
        )
        rows = cur.fetchall()
        if not rows:
            break

        for row_id, user_msg, asst_msg in rows:
            h = content_hash(user_msg or "", asst_msg or "")
            try:
                cur.execute(
                    "UPDATE all_records SET content_hash = ? WHERE id = ?",
                    (h, row_id),
                )
                updated += 1
            except sqlite3.IntegrityError:
                # Duplicate — mark for removal
                cur.execute("DELETE FROM all_records WHERE id = ?", (row_id,))
                updated += 1

        conn.commit()
        log.info(f"  Backfilled {updated:,} / {total:,}")

    # Create unique index if it doesn't exist
    try:
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_content_hash ON all_records(content_hash)")
        conn.commit()
    except sqlite3.IntegrityError:
        log.warning("Duplicate hashes found during index creation — run dedup first")

    return updated


# ============================================================================
# Statistics
# ============================================================================

def compute_stats(conn: sqlite3.Connection) -> None:
    """Compute and store domain statistics including quality metrics."""
    cur = conn.cursor()

    for domain in DOMAINS:
        try:
            cur.execute(f"SELECT COUNT(*) FROM {domain}")
            count = cur.fetchone()[0]
        except Exception:
            count = 0

        if count > 0:
            cur.execute(f"SELECT AVG(char_length) FROM {domain}")
            avg_len = cur.fetchone()[0] or 0

            cur.execute(f"SELECT COUNT(*) FROM {domain} WHERE cve_ids IS NOT NULL")
            total_cves = cur.fetchone()[0]

            cur.execute(f"SELECT COUNT(*) FROM {domain} WHERE mitre_techniques IS NOT NULL")
            total_techniques = cur.fetchone()[0]

            severity_counts = {}
            for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
                cur.execute(f"SELECT COUNT(*) FROM {domain} WHERE severity = ?", (sev,))
                severity_counts[sev] = cur.fetchone()[0]

            # Quality stats from master table
            cur.execute(
                "SELECT AVG(quality_score) FROM all_records WHERE domain = ? AND quality_score IS NOT NULL",
                (domain,),
            )
            avg_quality = cur.fetchone()[0]

            cur.execute(
                "SELECT COUNT(*) FROM all_records WHERE domain = ? AND validation_status IS NOT NULL",
                (domain,),
            )
            validated = cur.fetchone()[0]

            cur.execute(
                "SELECT COUNT(*) FROM all_records WHERE domain = ? AND validation_status = 'pass'",
                (domain,),
            )
            v_pass = cur.fetchone()[0]

            cur.execute(
                "SELECT COUNT(*) FROM all_records WHERE domain = ? AND validation_status = 'fail'",
                (domain,),
            )
            v_fail = cur.fetchone()[0]
        else:
            avg_len = 0
            total_cves = total_techniques = 0
            severity_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
            avg_quality = None
            validated = v_pass = v_fail = 0

        cur.execute("""
            INSERT OR REPLACE INTO domain_stats
                (domain, record_count, avg_length, total_cves, total_techniques,
                 severity_critical, severity_high, severity_medium, severity_low,
                 avg_quality_score, validated_count, validation_pass_count,
                 validation_fail_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (domain, count, avg_len, total_cves, total_techniques,
              severity_counts["CRITICAL"], severity_counts["HIGH"],
              severity_counts["MEDIUM"], severity_counts["LOW"],
              avg_quality, validated, v_pass, v_fail))

    conn.commit()


# ============================================================================
# Connection Helpers
# ============================================================================

def get_connection(db_path: str | Path | None = None, wal: bool = True) -> sqlite3.Connection:
    """Get a database connection with optimal settings."""
    path = str(db_path or DEFAULT_DB_PATH)
    conn = sqlite3.connect(path)
    if wal:
        conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def get_row_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Get a connection with row_factory set for dict-like access."""
    conn = get_connection(db_path)
    conn.row_factory = sqlite3.Row
    return conn


# ============================================================================
# Message Parsing Helpers
# ============================================================================

def parse_messages(record: dict) -> tuple[str, str]:
    """Extract user and assistant messages from a messages record.

    Handles both {"messages": [...]} and bare [...] formats.
    """
    msgs = record.get("messages", record) if isinstance(record, dict) else record
    user_msg = ""
    assistant_msg = ""
    for m in msgs:
        if m["role"] == "user":
            user_msg = m["content"]
        elif m["role"] == "assistant":
            assistant_msg = m["content"]
    return user_msg, assistant_msg
