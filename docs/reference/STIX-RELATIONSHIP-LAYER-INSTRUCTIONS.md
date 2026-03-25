# STIX Relationship Layer — Implementation Instructions

## Overview

Add a STIX 2.1 relationship graph layer to the existing MDR training database (`mdr-database/mdr_dataset.db`). This converts our flat 365K training examples into a connected knowledge graph where threat actors, techniques, vulnerabilities, malware, tools, and mitigations are linked by typed relationships. The graph enables automated training data generation through relationship traversal and powers the K-LLM synthesis layer.

## Why This Matters

The existing database has the **nodes** (365K training examples covering CVEs, APT intel, MITRE techniques, etc.) but is missing the **edges** (STIX relationships that connect them). Right now:
- CVE records and APT records sit in separate silos
- The `mitre_techniques` and `cve_ids` columns in `all_records` are just comma-separated text strings
- Asking "What techniques does APT28 use?" requires the model to have memorized that from training text

After this implementation:
- Every actor, technique, CVE, malware, tool, and mitigation is a STIX object node
- Typed relationships (uses, exploits, targets, mitigates, attributed-to) connect them
- Graph traversal generates thousands of new training pairs automatically
- Cross-domain queries become trivial: `CVE-2024-3400 ←[exploits]← T1190 ←[uses]← APT28`

---

## Existing Database Context

**Database:** `mdr-database/mdr_dataset.db` (SQLite, Schema v4)
**Records:** 365,452 training examples in `all_records` table
**Key existing sources that already contain STIX-mappable data:**

| Source | Records | Maps to STIX Type |
|---|---:|---|
| `mitre_stix_groups` | 1,667 | `intrusion-set` (threat actors) |
| `mitre_stix` | 1,382 | `attack-pattern` (techniques) |
| `misp_galaxy` | 1,611 | `intrusion-set` + `malware` + `tool` |
| `cveorg` | 345,489 | `vulnerability` |
| `cisa_kev` | 4,578 | `vulnerability` + `indicator` |
| `spl_production` | 5,533 | `course-of-action` (detection rules) |
| `sigma_rules` | 499 | `course-of-action` (detection rules) |

**Key existing columns in `all_records` that become graph edges:**
- `cve_ids` — comma-separated CVE IDs (currently text, should become relationships)
- `mitre_techniques` — comma-separated technique IDs (currently text, should become relationships)
- `cwe_ids` — comma-separated CWE IDs

**Existing source scripts (in `scripts/sources/`) to reference for patterns:**
- `fetch_mitre_stix.py` — already downloads ATT&CK STIX data but discards relationships
- `fetch_mitre_groups.py` — already downloads threat group data but discards relationships
- `fetch_misp_galaxy.py` — already downloads MISP actor intel
- `fetch_cisa_kev.py` — already downloads KEV catalog
- `ingest_cveorg.py` — already ingests CVE records

---

## Implementation Plan

### Phase 1: Schema Migration (v4 → v5)

Add three new tables to the existing SQLite database. This is a non-destructive migration — no existing tables or data are modified.

**File:** `scripts/migrate_v5_stix.py`

Create a migration script that:
1. Checks current schema version in `schema_info` table
2. Only runs if version < 5
3. Creates the three new tables
4. Updates schema_info to version 5

**SQL for the three new tables:**

```sql
-- STIX Objects (the nodes of the graph)
CREATE TABLE IF NOT EXISTS stix_objects (
    stix_id TEXT PRIMARY KEY,           -- Full STIX ID: "intrusion-set--abc-123-..."
    type TEXT NOT NULL,                  -- STIX type: intrusion-set, attack-pattern, malware, vulnerability, tool, course-of-action, campaign, identity, indicator
    name TEXT NOT NULL,                  -- Human-readable name: "APT28", "T1566", "CVE-2024-3400"
    aliases TEXT,                        -- JSON array of alternate names: ["Fancy Bear", "Sofacy", "Pawn Storm"]
    description TEXT,                    -- Full description text
    external_ids TEXT,                   -- JSON object of external references: {"mitre_attack_id": "G0007", "cve_id": "CVE-2024-3400"}
    source TEXT NOT NULL,               -- Which pipeline produced this: "mitre_attack", "misp_galaxy", "cveorg", "cisa_kev", "sigma_hq"
    platforms TEXT,                      -- JSON array: ["Windows", "Linux", "macOS"] (for techniques)
    kill_chain_phases TEXT,             -- JSON array of tactic phases (for techniques)
    severity TEXT,                       -- CRITICAL/HIGH/MEDIUM/LOW (for vulnerabilities)
    cvss_score REAL,                    -- CVSS score (for vulnerabilities)
    raw_stix_json TEXT,                 -- Full original STIX JSON object for reference
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

-- STIX Relationships (the edges of the graph)
CREATE TABLE IF NOT EXISTS stix_relationships (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    relationship_id TEXT UNIQUE,         -- STIX relationship ID if available: "relationship--abc-123-..."
    source_ref TEXT NOT NULL,            -- FK to stix_objects.stix_id (the "from" node)
    target_ref TEXT NOT NULL,            -- FK to stix_objects.stix_id (the "to" node)
    relationship_type TEXT NOT NULL,     -- "uses", "exploits", "targets", "mitigates", "attributed-to", "indicates", "variant-of", "delivers"
    description TEXT,                    -- Description of this specific relationship
    source TEXT NOT NULL,               -- Which pipeline produced this: "mitre_attack", "misp_galaxy", "cve_mapping", "sigma_mapping"
    confidence INTEGER,                 -- 0-100 confidence score if available
    created_at TEXT DEFAULT (datetime('now')),
    FOREIGN KEY (source_ref) REFERENCES stix_objects(stix_id),
    FOREIGN KEY (target_ref) REFERENCES stix_objects(stix_id)
);

-- Bridge table: links STIX objects to existing training examples in all_records
CREATE TABLE IF NOT EXISTS stix_training_links (
    stix_id TEXT NOT NULL,              -- FK to stix_objects.stix_id
    record_id INTEGER NOT NULL,         -- FK to all_records.id
    link_type TEXT NOT NULL,            -- "mentions", "about", "detects", "analyzes", "mitigates"
    confidence REAL DEFAULT 1.0,        -- How confident the link is (1.0 = exact match, 0.5 = fuzzy)
    created_at TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (stix_id, record_id),
    FOREIGN KEY (stix_id) REFERENCES stix_objects(stix_id),
    FOREIGN KEY (record_id) REFERENCES all_records(id)
);

-- Performance indexes for graph traversal
CREATE INDEX IF NOT EXISTS idx_stix_obj_type ON stix_objects(type);
CREATE INDEX IF NOT EXISTS idx_stix_obj_name ON stix_objects(name);
CREATE INDEX IF NOT EXISTS idx_stix_obj_source ON stix_objects(source);
CREATE INDEX IF NOT EXISTS idx_stix_rel_source ON stix_relationships(source_ref);
CREATE INDEX IF NOT EXISTS idx_stix_rel_target ON stix_relationships(target_ref);
CREATE INDEX IF NOT EXISTS idx_stix_rel_type ON stix_relationships(relationship_type);
CREATE INDEX IF NOT EXISTS idx_stix_rel_source_type ON stix_relationships(source_ref, relationship_type);
CREATE INDEX IF NOT EXISTS idx_stix_rel_target_type ON stix_relationships(target_ref, relationship_type);
CREATE INDEX IF NOT EXISTS idx_stix_link_stix ON stix_training_links(stix_id);
CREATE INDEX IF NOT EXISTS idx_stix_link_record ON stix_training_links(record_id);
```

**Also add a `stix_stats` table** for tracking ingestion metrics (same pattern as existing `domain_stats`):

```sql
CREATE TABLE IF NOT EXISTS stix_stats (
    type TEXT PRIMARY KEY,              -- STIX object type
    object_count INTEGER DEFAULT 0,
    relationship_count INTEGER DEFAULT 0,
    training_link_count INTEGER DEFAULT 0,
    last_updated TEXT
);
```

---

### Phase 2: STIX Object Ingestion Scripts

Create a new script that downloads the MITRE ATT&CK STIX bundle and extracts ALL object types and relationships — not just techniques and groups like the existing scripts do.

**File:** `scripts/sources/fetch_stix_objects.py`

This is the main ingestion script. It should:

#### Step 2a: Download the ATT&CK Enterprise STIX Bundle

```python
ATTACK_STIX_URL = "https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json"
```

Download to `data/sources/enterprise-attack.json` (cache locally, re-download if older than 7 days).

The bundle is a single JSON file with a `objects` array containing every ATT&CK object. Each object has a `type` field.

#### Step 2b: Parse and Store STIX Objects

Iterate through the `objects` array and extract objects of these types:

| STIX `type` value | What it is | Expected count (~) |
|---|---|---|
| `intrusion-set` | Threat actor groups (APT28, Lazarus, etc.) | ~150 |
| `attack-pattern` | Techniques and sub-techniques | ~800 |
| `malware` | Malware families | ~600 |
| `tool` | Legitimate tools used maliciously | ~80 |
| `campaign` | Named campaigns | ~30 |
| `course-of-action` | Mitigations | ~45 |
| `x-mitre-tactic` | Tactics (the columns in ATT&CK matrix) | ~14 |

For each object, extract:
- `stix_id`: the object's `id` field (e.g., `"intrusion-set--bef4c620-0787-42a8-a96d-b7eb6e85917c"`)
- `type`: the object's `type` field
- `name`: the object's `name` field
- `description`: the object's `description` field
- `aliases`: look for `aliases` field (intrusion-sets) or `x_mitre_aliases` field (malware/tools)
- `external_ids`: extract from `external_references` array — look for entries with `source_name` == `"mitre-attack"` and grab the `external_id` (e.g., `"G0007"`, `"T1566"`, `"S0154"`)
- `platforms`: from `x_mitre_platforms` if present
- `kill_chain_phases`: from `kill_chain_phases` array if present

**Important:** Skip objects where `revoked` is `true` or `x_mitre_deprecated` is `true`.

**STIX ID format:** Every STIX object has an `id` like `"attack-pattern--a93494bb-4b80-4ea1-8695-3236a49916fd"`. Use this as the `stix_id` primary key.

#### Step 2c: Parse and Store STIX Relationships

The bundle also contains objects with `type` == `"relationship"`. These are the edges.

Each relationship object has:
- `id`: relationship STIX ID
- `source_ref`: STIX ID of the source object
- `target_ref`: STIX ID of the target object
- `relationship_type`: string like `"uses"`, `"mitigates"`, `"attributed-to"`, `"targets"`, `"subtechnique-of"`, etc.
- `description`: sometimes present, describes the specific use

For each relationship:
1. Verify both `source_ref` and `target_ref` exist in `stix_objects` (skip orphan relationships)
2. Skip relationships where the source or target object was revoked/deprecated
3. Insert into `stix_relationships`

**Expected relationship types and counts (~):**

| relationship_type | Meaning | Count (~) |
|---|---|---|
| `uses` | Actor/malware/tool uses technique | ~12,000 |
| `mitigates` | Mitigation addresses technique | ~400 |
| `subtechnique-of` | Sub-technique parent link | ~400 |
| `attributed-to` | Campaign attributed to actor | ~30 |
| `targets` | Actor/campaign targets identity | ~50 |
| `revoked-by` | Replaced by newer object | varies |

#### Step 2d: Ingest MISP Galaxy Threat Actors

**File:** Extend `fetch_stix_objects.py` or create `scripts/sources/fetch_stix_misp.py`

Download the MISP Galaxy threat actor cluster:
```
https://raw.githubusercontent.com/MISP/misp-galaxy/main/clusters/threat-actor.json
```

Parse the `values` array. Each entry has:
- `value`: actor name (e.g., "APT28")
- `description`: detailed description
- `uuid`: MISP UUID
- `meta.synonyms`: array of aliases
- `meta.country`: attributed country
- `meta.refs`: reference URLs

For each MISP actor:
1. Generate a STIX-style ID: `"intrusion-set--{misp_uuid}"`
2. Check if this actor already exists in `stix_objects` (match by name or alias against MITRE ATT&CK intrusion-sets)
3. If it exists: merge any new aliases or description text (MISP often has more aliases than ATT&CK)
4. If it doesn't exist: insert as new `stix_objects` entry with `source = "misp_galaxy"`

**Important:** MISP has ~500 actors vs MITRE's ~150. Many overlap. Deduplicate by matching names and aliases. When merging, prefer MITRE's STIX ID as primary and add MISP UUID to `external_ids`.

#### Step 2e: Ingest CVEs as STIX Vulnerability Objects

Pull CVE IDs from the existing `all_records` table (the `cve_ids` column) and from the `cve` domain table if it exists.

For each unique CVE ID found:
1. Generate STIX ID: `"vulnerability--{uuid5 from CVE ID}"` (use `uuid.uuid5(uuid.NAMESPACE_URL, cve_id)`)
2. Pull description, severity, cvss_score from existing records
3. Insert into `stix_objects` with `type = "vulnerability"`, `source = "cveorg"` or `"cisa_kev"`

Then create relationships:
- If a CVE appears in a CISA KEV record → indicates it's actively exploited
- Cross-reference CVE mentions in MITRE ATT&CK technique descriptions to create `vulnerability ←[exploits]← attack-pattern` relationships

---

### Phase 3: Link STIX Objects to Training Records

**File:** `scripts/link_stix_training.py`

This script populates `stix_training_links` by scanning existing `all_records` and linking them to STIX objects.

#### Linking Strategy:

1. **CVE-based links:** For every record in `all_records` that has a non-empty `cve_ids` column:
   - Parse comma-separated CVE IDs
   - Find matching `stix_objects` entry (type=vulnerability)
   - Insert into `stix_training_links` with `link_type = "about"`, `confidence = 1.0`

2. **MITRE technique links:** For every record with non-empty `mitre_techniques`:
   - Parse comma-separated technique IDs (e.g., "T1566", "T1059.001")
   - Find matching `stix_objects` entry (type=attack-pattern, matching external_id)
   - Insert into `stix_training_links` with `link_type = "about"`, `confidence = 1.0`

3. **APT Intel links:** For records in the `apt_intel` domain:
   - Search `user_message` and `assistant_message` for known threat actor names and aliases
   - Match against `stix_objects` names and aliases (case-insensitive)
   - Insert with `link_type = "about"`, `confidence = 0.8` (text-based matching is slightly less certain)

4. **Detection rule links:** For records in `siem_queries` and `sigma_rules` domains:
   - If the record has `mitre_techniques`, link to those technique STIX objects
   - Set `link_type = "detects"`

5. **Log analysis links:** For records in `log_analysis` domain:
   - Link to techniques via `mitre_techniques` column
   - Set `link_type = "analyzes"`

**Performance note:** Process in batches of 1000 records. Use `INSERT OR IGNORE` to handle re-runs without duplicating links.

---

### Phase 4: Graph Query Utilities

**File:** `scripts/stix_graph.py`

Create a utility module with graph traversal functions. These will be used by both the training pair generator and any future RAG/query pipeline.

**Required functions:**

```python
def get_object_by_name(db_path: str, name: str) -> dict:
    """Find a STIX object by name or alias (case-insensitive)."""

def get_relationships(db_path: str, stix_id: str, direction: str = "both", rel_type: str = None) -> list[dict]:
    """Get all relationships for a STIX object.
    direction: 'outgoing' (this object is source), 'incoming' (this object is target), 'both'
    rel_type: filter by relationship type (e.g., 'uses', 'mitigates')
    Returns list of dicts with relationship info and connected object details.
    """

def get_connected_objects(db_path: str, stix_id: str, rel_type: str = None, target_type: str = None) -> list[dict]:
    """Get all objects connected to this one, optionally filtered by relationship and object type.
    Example: get_connected_objects(db, apt28_id, rel_type="uses", target_type="attack-pattern")
    → returns all techniques APT28 uses
    """

def traverse_path(db_path: str, start_id: str, path_spec: list[tuple]) -> list[dict]:
    """Multi-hop graph traversal.
    path_spec is a list of (relationship_type, target_object_type) tuples.
    Example: traverse_path(db, apt28_id, [("uses", "attack-pattern"), ("exploits", "vulnerability")])
    → returns all vulnerabilities exploited by techniques that APT28 uses
    """

def get_subgraph(db_path: str, stix_id: str, depth: int = 2) -> dict:
    """Get the full subgraph around a node up to N hops.
    Returns {"nodes": [...], "edges": [...]} suitable for visualization.
    """

def get_training_records_for_object(db_path: str, stix_id: str) -> list[dict]:
    """Get all training records linked to a STIX object via stix_training_links."""

def get_stix_objects_for_record(db_path: str, record_id: int) -> list[dict]:
    """Get all STIX objects linked to a training record."""

def get_stats(db_path: str) -> dict:
    """Return counts: objects by type, relationships by type, training links by type."""
```

**Also create a CLI interface** so the graph can be queried from the command line:

```bash
# Show all techniques used by APT28
python scripts/stix_graph.py --query actor-techniques --name "APT28"

# Show all actors that exploit a specific CVE
python scripts/stix_graph.py --query cve-actors --cve "CVE-2024-3400"

# Show full subgraph around an object (2 hops)
python scripts/stix_graph.py --query subgraph --name "APT28" --depth 2

# Show database stats
python scripts/stix_graph.py --stats
```

---

### Phase 5: Training Data Generator

**File:** `scripts/generate_stix_training_pairs.py`

This is the high-value deliverable. It traverses the STIX relationship graph to automatically generate instruction-tuning pairs that teach the model to reason about relationships between threat intelligence objects.

**Generate these categories of training pairs:**

#### Category 1: Actor → Technique Mapping (~5K pairs)

For each `intrusion-set` object:
```
Q: "What MITRE ATT&CK techniques does {actor_name} use?"
A: "{actor_name} ({aliases}) is known to use the following techniques:
    - {T_ID} {technique_name}: {relationship_description or technique_description}
    - ...
    These techniques span the following tactics: {list of unique tactics from kill_chain_phases}"
```

Variations:
- "Which initial access techniques has {actor} been observed using?"
- "Describe {actor}'s command and control methods"
- "What are the TTPs associated with {actor}?"

#### Category 2: Actor → Malware/Tool Arsenal (~3K pairs)

```
Q: "What malware and tools does {actor_name} use?"
A: "{actor_name} has been associated with the following:
    Malware: {list of malware with descriptions}
    Tools: {list of tools with descriptions}
    {malware_name} is used for {relationship_description}..."
```

#### Category 3: Technique → Actor Attribution (~3K pairs)

Reverse direction — given a technique, who uses it:
```
Q: "Which threat actors have been observed using {technique_id} ({technique_name})?"
A: "The following groups have been observed using {technique_name}:
    - {actor_name} ({country attribution}): {relationship_description}
    - ...
    This technique is commonly used in {tactic} phase of attacks."
```

#### Category 4: Vulnerability → Kill Chain (~3K pairs)

Multi-hop traversal: vulnerability → technique → actor
```
Q: "Analyze CVE-{id} from a threat intelligence perspective. Which actors exploit it and through what techniques?"
A: "{CVE description}. CVSS: {score}.
    This vulnerability is exploited via:
    - {technique_id} ({technique_name}): {description}
    Known threat actors exploiting this vulnerability:
    - {actor_name}: uses {technique} which exploits this CVE
    Recommended mitigations:
    - {mitigation_name}: {description}"
```

#### Category 5: Full Kill Chain Narratives (~2K pairs)

Select an actor and trace through their full attack graph:
```
Q: "Describe the full attack lifecycle of {actor_name}, including initial access, execution, persistence, and impact."
A: "## {actor_name} Attack Lifecycle
    
    **Initial Access:** {actor} gains initial access through {techniques with rel_type uses, filtered to initial-access tactic}...
    **Execution:** Once inside, they leverage {execution techniques}...
    **Persistence:** {persistence techniques and associated malware}...
    **Collection & Exfiltration:** {collection/exfil techniques}...
    **Tools Used:** {tools and malware from uses relationships}
    **Known Campaigns:** {campaigns from attributed-to relationships}
    **Target Sectors:** {identities from targets relationships}"
```

#### Category 6: Detection → Attribution (~2K pairs)

Connect detection rules (from Sigma/SPL records) to threat actors:
```
Q: "A Sigma rule triggered for {technique_id} ({technique_name}). What threat actors should I investigate?"
A: "This technique is associated with the following threat actors:
    {list of actors from uses relationships}
    
    Priority investigation: {actors known to target your region/sector}
    
    Additional IOCs to hunt for: {indicators linked to these actors}
    Related techniques often used in conjunction: {techniques commonly used by same actors}"
```

#### Category 7: Mitigation Mapping (~1K pairs)

```
Q: "What mitigations are effective against {technique_id} ({technique_name})?"
A: "The following mitigations address {technique_name}:
    - {mitigation_name}: {description}
    
    This technique is used by: {list of actors}
    Priority: {HIGH if used by many actors or in CISA KEV}"
```

**Implementation requirements:**
- Each pair must have `domain`, `question_type`, `user_message`, `assistant_message` fields matching the `all_records` schema
- Deduplicate using the same content_hash approach as existing pipeline
- Run validation using existing `validate_data.py` patterns
- Output as JSONL to `data/sources/stix_graph_pairs.jsonl`
- Also support direct ingestion into the database via `ingest_data.py`

**Quality controls:**
- Only generate pairs from objects that have 2+ relationships (isolated nodes produce low-quality pairs)
- Vary question phrasing using templates (at least 5 phrasings per category)
- Include both "broad" questions (all techniques for an actor) and "narrow" questions (specific technique + specific actor)
- Cap assistant_message length at ~2000 chars to match existing training data distribution
- Tag all generated records with `source = "stix_graph"` for tracking

---

### Phase 6: Integration & Verification

#### 6a: Run Full Pipeline

Create a master script `scripts/build_stix_graph.py` that orchestrates everything:

```bash
python scripts/build_stix_graph.py
```

This should run in order:
1. `migrate_v5_stix.py` — Create tables if needed
2. `fetch_stix_objects.py` — Download and ingest ATT&CK + MISP objects and relationships
3. `link_stix_training.py` — Link existing 365K records to STIX objects
4. `generate_stix_training_pairs.py` — Generate new training pairs from graph
5. Print summary stats

#### 6b: Verification Queries

After the pipeline runs, verify with these queries:

```sql
-- Total STIX objects by type
SELECT type, COUNT(*) FROM stix_objects GROUP BY type ORDER BY COUNT(*) DESC;
-- Expected: attack-pattern ~800, malware ~600, intrusion-set ~500+, tool ~80, etc.

-- Total relationships by type
SELECT relationship_type, COUNT(*) FROM stix_relationships GROUP BY relationship_type ORDER BY COUNT(*) DESC;
-- Expected: uses ~12000, mitigates ~400, subtechnique-of ~400, etc.

-- Training links by type
SELECT link_type, COUNT(*) FROM stix_training_links GROUP BY link_type;
-- Expected: about ~350K+ (from CVE and technique matching), detects ~6K, analyzes ~1K

-- Verify APT28 has relationships
SELECT so2.name, sr.relationship_type, so2.type
FROM stix_relationships sr
JOIN stix_objects so1 ON sr.source_ref = so1.stix_id
JOIN stix_objects so2 ON sr.target_ref = so2.stix_id
WHERE so1.name = 'APT28'
LIMIT 20;

-- Verify cross-domain links work
SELECT so.name, so.type, COUNT(stl.record_id) as training_records
FROM stix_objects so
JOIN stix_training_links stl ON so.stix_id = stl.stix_id
GROUP BY so.stix_id
ORDER BY training_records DESC
LIMIT 20;
```

#### 6c: Update domain_stats

After ingestion, update the `domain_stats` table with a new entry for the `stix_graph` domain showing record counts, and update `stix_stats` with object/relationship/link counts per type.

---

## File Summary

| File | Purpose |
|---|---|
| `scripts/migrate_v5_stix.py` | Schema migration v4 → v5, creates 4 new tables |
| `scripts/sources/fetch_stix_objects.py` | Downloads ATT&CK STIX bundle + MISP galaxy, ingests all objects and relationships |
| `scripts/link_stix_training.py` | Links existing 365K records to STIX objects via CVE IDs, technique IDs, actor names |
| `scripts/stix_graph.py` | Graph query utility library + CLI for traversal queries |
| `scripts/generate_stix_training_pairs.py` | Generates 15-20K new training pairs from graph traversal |
| `scripts/build_stix_graph.py` | Master orchestrator — runs everything in sequence |

---

## Critical Implementation Notes

1. **Do NOT modify existing tables.** The `all_records` table and all domain tables must remain untouched. The STIX layer is purely additive.

2. **Use the existing project patterns.** Look at `scripts/sources/fetch_mitre_stix.py` and `scripts/db_utils.py` for how the project handles database connections, logging, and error handling. Follow the same patterns.

3. **SQLite concurrency.** Use `WAL` mode and proper connection handling. The database may be accessed by other scripts during development.

4. **Network access.** The script needs to download from:
   - `https://raw.githubusercontent.com/mitre-attack/attack-stix-data/master/enterprise-attack/enterprise-attack.json` (~30MB)
   - `https://raw.githubusercontent.com/MISP/misp-galaxy/main/clusters/threat-actor.json` (~2MB)
   
   Cache downloads locally in `data/sources/` and only re-download if the cached file is older than 7 days.

5. **Idempotent runs.** All scripts must be safe to run multiple times. Use `INSERT OR REPLACE` for stix_objects, `INSERT OR IGNORE` for relationships and links. Running the pipeline twice should not create duplicates.

6. **Memory efficiency.** The ATT&CK STIX bundle is ~30MB JSON. Parse it as a stream or load once and iterate — don't load multiple copies. The CVE linking step iterates 350K+ records — process in batches.

7. **STIX object deduplication between sources.** MITRE ATT&CK and MISP Galaxy both describe threat actors. When the same actor appears in both:
   - Use MITRE's STIX ID as primary
   - Merge MISP's additional aliases into the aliases JSON array
   - Keep both descriptions (concatenate with source labels)
   - Match by: exact name match, or any alias in one source matching name/alias in the other

8. **Relationship direction matters.** In STIX:
   - `intrusion-set --[uses]--> attack-pattern` (actor uses technique)
   - `intrusion-set --[uses]--> malware` (actor uses malware)
   - `malware --[uses]--> attack-pattern` (malware implements technique)
   - `course-of-action --[mitigates]--> attack-pattern` (mitigation addresses technique)
   - `attack-pattern --[subtechnique-of]--> attack-pattern` (sub-technique relationship)
   - `campaign --[attributed-to]--> intrusion-set` (campaign attributed to actor)
   
   Preserve the original direction from the STIX data. Do not reverse relationships.

9. **Expected final counts after full pipeline:**
   - `stix_objects`: ~2,000-2,500 (ATT&CK objects + MISP actors + CVE objects)
   - `stix_relationships`: ~13,000-15,000 (mostly "uses" from ATT&CK)
   - `stix_training_links`: ~350,000+ (most from CVE ID matching)
   - New training pairs generated: ~15,000-20,000

---

## Testing

After implementation, run these smoke tests:

```bash
# 1. Check schema migration
python scripts/migrate_v5_stix.py
sqlite3 mdr-database/mdr_dataset.db ".tables" | grep stix

# 2. Check STIX ingestion
python scripts/sources/fetch_stix_objects.py
python scripts/stix_graph.py --stats

# 3. Test graph queries
python scripts/stix_graph.py --query actor-techniques --name "APT28"
python scripts/stix_graph.py --query actor-techniques --name "Lazarus Group"
python scripts/stix_graph.py --query cve-actors --cve "CVE-2024-3400"

# 4. Check training links
python scripts/link_stix_training.py
sqlite3 mdr-database/mdr_dataset.db "SELECT link_type, COUNT(*) FROM stix_training_links GROUP BY link_type;"

# 5. Generate training pairs
python scripts/generate_stix_training_pairs.py
wc -l data/sources/stix_graph_pairs.jsonl

# 6. Full pipeline
python scripts/build_stix_graph.py
```
