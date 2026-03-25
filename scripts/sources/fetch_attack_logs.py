#!/usr/bin/env python3
"""
============================================================================
Attack Log Training Data Generator
============================================================================
Generates detection engineering training examples from the MDR Log Simulator
API across three sub-domains:

  - log_analysis:  Raw log triage and attack identification
  - siem_queries:  SPL/KQL detection query writing
  - sigma_rules:   Sigma detection rule writing and explanation

Uses the simulator's /api/mitre/preview/{technique_id} endpoint to get
coordinated multi-source attack logs with ground-truth labels, then strips
those labels for question-side logs so the model learns analysis, not
label-reading.

Usage:
    python -m scripts.sources.fetch_attack_logs
    python -m scripts.sources.fetch_attack_logs --with-sigma-hq
    python -m scripts.sources.fetch_attack_logs --output data/sources/attack_logs.jsonl
============================================================================
"""

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import URLError

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "sources" / "attack_logs.jsonl"

# Ground-truth fields to strip from question-side logs
GROUND_TRUTH_FIELDS = {
    "mitre_technique", "mitre_tactic", "severity", "threat_name",
    "detection_source", "threat_classification", "confidence",
    "mitre_technique_name",
}

# Map generator names to human-readable log source names
GENERATOR_DISPLAY_NAMES = {
    "endpoint_generator": "Endpoint Security",
    "microsoft_defender": "Microsoft Defender for Endpoint",
    "crowdstrike": "CrowdStrike Falcon",
    "sentinelone": "SentinelOne",
    "palo_alto": "Palo Alto Networks",
    "fortinet": "FortiGate",
    "cisco_asa": "Cisco ASA",
    "checkpoint": "Check Point",
    "zscaler": "Zscaler",
    "okta": "Okta",
    "duo": "Duo Security",
    "aws_cloudtrail": "AWS CloudTrail",
    "azure_activity": "Azure Activity Log",
    "office365": "Microsoft 365",
    "windows_event": "Windows Event Log",
    "linux_audit": "Linux Audit",
    "sophos": "Sophos",
    "carbon_black": "VMware Carbon Black",
    "windows_performance_generator": "Windows Performance Metrics",
}


# ============================================================================
# API Client
# ============================================================================

class SimulatorClient:
    """Lightweight client for the MDR Log Simulator API."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self._benign_cache: dict[str, list[dict]] = {}

    def _get(self, path: str, retries: int = 3) -> dict | list:
        url = f"{self.base_url}{path}"
        for attempt in range(retries):
            try:
                req = Request(url, headers={"User-Agent": "MDR-Training-Pipeline/1.0"})
                with urlopen(req, timeout=30) as resp:
                    return json.loads(resp.read().decode())
            except URLError as e:
                if "429" in str(e) and attempt < retries - 1:
                    delay = 2 ** (attempt + 1)
                    time.sleep(delay)
                    continue
                raise

    def _post(self, path: str, body: dict, retries: int = 3) -> dict:
        url = f"{self.base_url}{path}"
        for attempt in range(retries):
            try:
                data = json.dumps(body).encode()
                req = Request(url, data=data, method="POST", headers={
                    "User-Agent": "MDR-Training-Pipeline/1.0",
                    "Content-Type": "application/json",
                })
                with urlopen(req, timeout=30) as resp:
                    return json.loads(resp.read().decode())
            except URLError as e:
                if "429" in str(e) and attempt < retries - 1:
                    delay = 2 ** (attempt + 1)
                    time.sleep(delay)
                    continue
                raise

    def get_techniques(self) -> list[dict]:
        """Fetch all mapped MITRE techniques."""
        return self._get("/api/mitre/techniques?mapped_only=true")

    def get_preview(self, technique_id: str) -> dict:
        """Fetch preview logs for a technique."""
        return self._get(f"/api/mitre/preview/{technique_id}")

    def get_generators(self, technique_id: str) -> list[dict]:
        """Fetch generator metadata for a technique."""
        return self._get(f"/api/mitre/techniques/{technique_id}/generators")

    def get_benign_logs(self, log_type: str, count: int = 5) -> list[dict]:
        """Fetch benign baseline logs, cached by log_type."""
        if log_type in self._benign_cache:
            return self._benign_cache[log_type]
        try:
            resp = self._post("/api/v1/logs/generate", {
                "log_type": log_type,
                "count": count,
            })
            events = resp.get("events", [])
            self._benign_cache[log_type] = events
            return events
        except (URLError, json.JSONDecodeError) as e:
            log.debug(f"Failed to get benign {log_type} logs: {e}")
            return []


# ============================================================================
# Log Sanitization
# ============================================================================

def sanitize_log(event: dict) -> dict:
    """Strip ground-truth fields from a log event for question embedding."""
    return {k: v for k, v in event.items() if k not in GROUND_TRUTH_FIELDS}


def format_log_json(event: dict, compact: bool = False) -> str:
    """Format a log event as a JSON string for embedding in questions."""
    if compact:
        # Keep only key diagnostic fields for multi-event scenarios
        key_fields = [
            "timestamp", "action", "host", "user", "src_ip", "dest_ip",
            "process_name", "parent_process_name", "command_line",
            "file_path", "event_id", "sourcetype", "task_name",
        ]
        compact_event = {k: v for k, v in event.items() if k in key_fields and v}
        return json.dumps(compact_event, indent=2)
    return json.dumps(event, indent=2)


def generator_display_name(gen_name: str) -> str:
    """Get human-readable name for a generator."""
    return GENERATOR_DISPLAY_NAMES.get(gen_name, gen_name.replace("_", " ").title())


# ============================================================================
# Template Selection
# ============================================================================

def _select_template(templates: list[str], key: str) -> str:
    """Deterministically select a template variant based on key hash."""
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return templates[h % len(templates)]


# ============================================================================
# Question Templates
# ============================================================================

# -- Log Analysis Templates --

SINGLE_SOURCE_TEMPLATES = [
    "Analyze the following {source_name} log entry and identify any malicious activity. "
    "If an attack is detected, specify the MITRE ATT&CK technique, describe the evidence, "
    "assess the severity, and recommend response actions.\n\n```json\n{log}\n```",

    "You are an MDR analyst. Triage this {source_name} event and determine if it represents "
    "a security threat. Provide the technique ID, key evidence fields, severity level, and "
    "recommended next steps.\n\n```json\n{log}\n```",

    "Analyze this {source_name} security event for indicators of compromise. Identify the "
    "attack technique, explain what specific fields indicate malicious behavior, rate the "
    "severity, and suggest immediate response actions.\n\n```json\n{log}\n```",
]

MULTI_SOURCE_TEMPLATES = [
    "The following correlated security events were collected from {source_count} different "
    "sources within a short time window. Analyze them together to identify the attack, map "
    "it to the MITRE ATT&CK framework, and describe the attack chain.\n\n{logs_block}",

    "You are correlating alerts from {source_count} security platforms. Analyze these events "
    "together, identify the MITRE ATT&CK technique being used, explain how the events are "
    "related, and provide a severity assessment with response recommendations.\n\n{logs_block}",

    "As an MDR analyst, correlate these {source_count} security events from different sources. "
    "What attack technique is being executed? How do the events corroborate each other? What "
    "is the overall severity and what actions should be taken?\n\n{logs_block}",
]

TRIAGE_TEMPLATES = [
    "Review the following {event_count} {source_name} events. Most are routine, but some may "
    "indicate malicious activity. Identify which events are suspicious, explain why, and "
    "classify the threat.\n\n{logs_block}",

    "You are triaging a batch of {event_count} {source_name} events. Separate benign activity "
    "from potential threats. For any suspicious events, identify the MITRE ATT&CK technique "
    "and provide a severity assessment.\n\n{logs_block}",

    "Analyze this batch of {event_count} {source_name} events and identify any that require "
    "investigation. Explain what distinguishes the malicious events from normal operations "
    "and recommend appropriate response actions.\n\n{logs_block}",
]

# -- SIEM Query Templates --

SPL_TEMPLATES = [
    "Write a Splunk SPL query to detect MITRE ATT&CK technique {technique_id} ({technique_name}) "
    "in {source_name} logs. The logs use sourcetype={sourcetype}. Include relevant field filters "
    "and explain the detection logic.",

    "As a detection engineer, create a Splunk SPL detection rule for {technique_id} "
    "({technique_name}). Use the {source_name} data source with sourcetype={sourcetype}. "
    "Include the query, explain what it detects, and suggest a recommended alert schedule.",

    "Write a Splunk search to identify {technique_name} ({technique_id}) activity. "
    "Target {source_name} logs (sourcetype={sourcetype}). Provide the SPL query, "
    "detection rationale, and recommended scheduling interval.",
]

CORRELATION_SPL_TEMPLATES = [
    "Write a Splunk correlation search that detects {technique_id} ({technique_name}) by "
    "joining events from these sources: {source_list}. Use subsearches or stats commands "
    "to correlate across sourcetypes.",

    "Create a Splunk SPL correlation rule for detecting {technique_name} ({technique_id}) "
    "across multiple data sources: {source_list}. The query should correlate events by "
    "host or user to reduce false positives.",
]

KQL_TEMPLATES = [
    "Write a KQL query for Microsoft Sentinel to detect MITRE ATT&CK technique {technique_id} "
    "({technique_name}). Use appropriate table names and field filters. Explain the detection "
    "logic and recommend a scheduling interval.",

    "As a detection engineer, create a Microsoft Sentinel KQL analytics rule to detect "
    "{technique_name} ({technique_id}). Include the KQL query, detection explanation, "
    "severity classification, and MITRE ATT&CK mapping.",
]

# -- Sigma Rule Templates --

SIGMA_WRITE_TEMPLATES = [
    "Write a Sigma detection rule to detect MITRE ATT&CK technique {technique_id} "
    "({technique_name}) based on the following log pattern:\n\n```json\n{log}\n```\n\n"
    "Include title, status, description, logsource, detection logic, false positives, "
    "and level.",

    "Create a Sigma rule that detects {technique_name} ({technique_id}). Use this sample "
    "log event as reference for the detection fields:\n\n```json\n{log}\n```\n\n"
    "Provide a complete Sigma rule with proper YAML structure.",

    "You are a detection engineer. Write a Sigma rule for {technique_id} ({technique_name}) "
    "based on this log evidence:\n\n```json\n{log}\n```\n\n"
    "The rule should include appropriate logsource, selection criteria, condition, "
    "and ATT&CK tags.",
]

SIGMA_EXPLAIN_TEMPLATES = [
    "Explain the following Sigma detection rule. Describe what it detects, how the "
    "detection logic works, what MITRE ATT&CK technique it maps to, and potential "
    "false positives.\n\n```yaml\n{rule}\n```",

    "As an MDR analyst, analyze this Sigma rule. Explain its purpose, walk through "
    "the detection conditions, identify the targeted ATT&CK technique, and assess "
    "its effectiveness.\n\n```yaml\n{rule}\n```",

    "What does this Sigma detection rule do? Break down each section, explain the "
    "detection logic, map it to the MITRE ATT&CK framework, and discuss when it "
    "might produce false positives.\n\n```yaml\n{rule}\n```",
]


# ============================================================================
# Answer Composers
# ============================================================================

def compose_single_source_answer(technique_id: str, technique_name: str,
                                  tactic: str, event: dict,
                                  source_name: str) -> str:
    """Compose an answer for single-source log analysis."""
    severity = str(event.get("severity", "medium"))
    parts = [
        f"**Attack Identified:** {technique_id} — {technique_name}",
        f"**MITRE ATT&CK Tactic:** {tactic.replace('-', ' ').title()}",
        f"**Severity:** {severity.upper()}",
        f"**Source:** {source_name}",
        "",
        "**Evidence:**",
    ]

    # Build evidence from key fields
    evidence_fields = {
        "command_line": "Command Line",
        "process_name": "Process",
        "parent_process_name": "Parent Process",
        "action": "Action",
        "task_name": "Task Name",
        "file_path": "File Path",
        "src_ip": "Source IP",
        "dest_ip": "Destination IP",
        "user": "User",
        "event_id": "Event ID",
    }
    for field, label in evidence_fields.items():
        if field in event and event[field]:
            parts.append(f"- **{label}:** `{event[field]}`")

    parts.extend([
        "",
        "**Analysis:**",
        _compose_technique_analysis(technique_id, technique_name, tactic, event),
        "",
        "**Recommended Response Actions:**",
    ])
    parts.extend(_compose_response_actions(technique_id, tactic, severity))

    return "\n".join(parts)


def compose_multi_source_answer(technique_id: str, technique_name: str,
                                 tactic: str, events: list[dict],
                                 sources: list[str]) -> str:
    """Compose an answer for multi-source correlation analysis."""
    parts = [
        f"**Correlated Attack:** {technique_id} — {technique_name}",
        f"**MITRE ATT&CK Tactic:** {tactic.replace('-', ' ').title()}",
        f"**Sources Correlated:** {', '.join(sources)}",
        f"**Severity:** HIGH (multi-source corroboration increases confidence)",
        "",
        "**Correlation Analysis:**",
        f"Events from {len(sources)} independent security platforms confirm "
        f"the execution of {technique_name} ({technique_id}):",
        "",
    ]

    for i, (event, source) in enumerate(zip(events, sources), 1):
        action = event.get("action", "unknown")
        host = event.get("host", "unknown")
        parts.append(f"{i}. **{source}** detected `{action}` on host `{host}`")
        if "command_line" in event:
            parts.append(f"   - Command: `{event['command_line'][:120]}`")
        if "process_name" in event:
            parts.append(f"   - Process: `{event['process_name']}`")

    parts.extend([
        "",
        "**Attack Chain Assessment:**",
        _compose_technique_analysis(technique_id, technique_name, tactic, events[0]),
        "",
        "**Recommended Response Actions:**",
    ])
    parts.extend(_compose_response_actions(technique_id, tactic, "high"))

    return "\n".join(parts)


def compose_triage_answer(technique_id: str, technique_name: str,
                           tactic: str, malicious_event: dict,
                           malicious_idx: int, total_events: int,
                           source_name: str) -> str:
    """Compose an answer for benign/malicious triage scenarios."""
    severity = str(malicious_event.get("severity", "medium"))
    parts = [
        f"**Triage Result:** {total_events - 1} benign events, 1 suspicious event detected.",
        "",
        f"**Suspicious Event (Event #{malicious_idx + 1}):**",
        f"- **Attack:** {technique_id} — {technique_name}",
        f"- **Tactic:** {tactic.replace('-', ' ').title()}",
        f"- **Severity:** {severity.upper()}",
        "",
        "**Why This Event Is Suspicious:**",
    ]

    # Explain what makes it different from benign events
    if "command_line" in malicious_event:
        parts.append(f"- Suspicious command line: `{malicious_event['command_line'][:150]}`")
    if "action" in malicious_event:
        parts.append(f"- Action type `{malicious_event['action']}` indicates potential attack behavior")
    if "process_name" in malicious_event:
        parts.append(f"- Process `{malicious_event['process_name']}` "
                     f"executing from unusual context")

    parts.extend([
        "",
        f"The remaining {total_events - 1} events represent normal {source_name} operations.",
        "",
        "**Recommended Response Actions:**",
    ])
    parts.extend(_compose_response_actions(technique_id, tactic, severity))

    return "\n".join(parts)


def compose_spl_answer(technique_id: str, technique_name: str,
                        tactic: str, event: dict,
                        sourcetype: str, source_name: str) -> str:
    """Compose a SPL query answer."""
    query = _build_spl_query(technique_id, technique_name, event, sourcetype)
    parts = [
        f"**Detection Rule for {technique_id} — {technique_name}**",
        "",
        "```spl",
        query,
        "```",
        "",
        "**Detection Logic:**",
        f"This query targets {source_name} logs (sourcetype=`{sourcetype}`) to identify "
        f"{technique_name} activity associated with the {tactic.replace('-', ' ').title()} tactic.",
        "",
    ]

    # Explain key field filters
    parts.append("**Key Field Filters:**")
    if "process_name" in event:
        parts.append(f"- `process_name`: Filters for `{event['process_name']}` execution")
    if "command_line" in event:
        parts.append(f"- `command_line`: Pattern matches suspicious command patterns")
    if "action" in event:
        parts.append(f"- `action`: Focuses on `{event['action']}` event types")
    if "event_id" in event:
        parts.append(f"- `event_id`: Targets Windows Event ID `{event['event_id']}`")

    parts.extend([
        "",
        f"**MITRE ATT&CK Mapping:** {technique_id} ({tactic.replace('-', ' ').title()})",
        "",
        "**Recommended Schedule:** Run every 5 minutes for near-real-time detection.",
    ])

    return "\n".join(parts)


def compose_correlation_spl_answer(technique_id: str, technique_name: str,
                                    tactic: str, events: list[dict],
                                    sourcetypes: list[str],
                                    sources: list[str]) -> str:
    """Compose a correlation SPL query answer."""
    query = _build_correlation_spl(technique_id, events, sourcetypes)
    parts = [
        f"**Correlation Detection for {technique_id} — {technique_name}**",
        "",
        "```spl",
        query,
        "```",
        "",
        "**Correlation Logic:**",
        f"This correlation search joins events across {len(sources)} data sources "
        f"({', '.join(sources)}) to detect {technique_name} with higher confidence.",
        "",
        "**How It Works:**",
        "1. The base search collects events from all relevant sourcetypes",
        f"2. Events are correlated by `host` and time window (±5 minutes)",
        f"3. The `stats` command requires activity in at least {min(len(sources), 2)} "
        f"sources to reduce false positives",
        "",
        f"**MITRE ATT&CK Mapping:** {technique_id} ({tactic.replace('-', ' ').title()})",
        "",
        "**Recommended Schedule:** Run every 15 minutes (correlation searches are resource-intensive).",
    ]

    return "\n".join(parts)


def compose_kql_answer(technique_id: str, technique_name: str,
                        tactic: str, event: dict,
                        source_name: str) -> str:
    """Compose a KQL query answer."""
    query = _build_kql_query(technique_id, technique_name, event)
    parts = [
        f"**Microsoft Sentinel Analytics Rule for {technique_id} — {technique_name}**",
        "",
        "```kql",
        query,
        "```",
        "",
        "**Detection Logic:**",
        f"This KQL query detects {technique_name} activity in Microsoft Sentinel. "
        f"It targets the {tactic.replace('-', ' ').title()} tactic phase.",
        "",
        "**Key Detection Fields:**",
    ]

    if "process_name" in event:
        parts.append(f"- Filters for process `{event['process_name']}`")
    if "command_line" in event:
        parts.append(f"- Pattern matches command-line arguments")
    if "action" in event:
        parts.append(f"- Focuses on `{event['action']}` event types")

    parts.extend([
        "",
        f"**MITRE ATT&CK Mapping:** {technique_id} ({tactic.replace('-', ' ').title()})",
        f"**Severity:** Medium-High",
        "",
        "**Recommended Schedule:** Run every 5 minutes with a 1-hour lookback window.",
    ])

    return "\n".join(parts)


def compose_sigma_write_answer(technique_id: str, technique_name: str,
                                tactic: str, event: dict) -> str:
    """Compose a Sigma rule writing answer."""
    rule = _build_sigma_rule(technique_id, technique_name, tactic, event)
    parts = [
        f"Here is a Sigma detection rule for {technique_id} — {technique_name}:",
        "",
        "```yaml",
        rule,
        "```",
        "",
        "**Rule Explanation:**",
        f"- **Title:** Detects {technique_name} activity based on observed log patterns",
        f"- **Logsource:** Configured for the appropriate log category and product",
        f"- **Detection:** Uses field-level matching on key indicators observed in the sample event",
        f"- **ATT&CK Mapping:** Tagged with `attack.{tactic}` and `attack.{technique_id.lower()}`",
        "",
        "**False Positive Considerations:**",
    ]

    # Add context-specific false positive notes
    if "schtasks" in str(event.get("command_line", "")).lower():
        parts.append("- Legitimate scheduled task creation by system administrators")
        parts.append("- Software deployment tools that use scheduled tasks")
    elif "powershell" in str(event.get("process_name", "")).lower():
        parts.append("- Administrative PowerShell scripts run by IT staff")
        parts.append("- Configuration management tools using PowerShell")
    else:
        parts.append("- Legitimate administrative tools performing similar operations")
        parts.append("- Automated system maintenance tasks")

    return "\n".join(parts)


def compose_sigma_explain_answer(rule_yaml: str, technique_id: str) -> str:
    """Compose an explanation of an existing Sigma rule."""
    # Parse basic fields from the YAML
    title = _yaml_field(rule_yaml, "title") or "Unknown"
    description = _yaml_field(rule_yaml, "description") or ""
    level = _yaml_field(rule_yaml, "level") or "medium"
    status = _yaml_field(rule_yaml, "status") or "unknown"

    parts = [
        f"**Rule: {title}**",
        "",
        f"**Purpose:** {description[:300]}" if description else "**Purpose:** Detection rule for security threats",
        "",
        "**Detection Logic Breakdown:**",
    ]

    # Parse detection section
    in_detection = False
    detection_lines = []
    for line in rule_yaml.splitlines():
        stripped = line.strip()
        if stripped.startswith("detection:"):
            in_detection = True
            continue
        if in_detection:
            if stripped and not stripped.startswith("#") and not stripped.startswith("-") and ":" in stripped and not line.startswith(" "):
                break
            detection_lines.append(line)

    if detection_lines:
        parts.append("```yaml")
        parts.extend(detection_lines[:15])
        parts.append("```")
        parts.append("")

    # Explain the detection
    parts.extend([
        "The rule works by matching specific field values in the log source. "
        "When the `condition` evaluates to true, an alert is generated.",
        "",
        f"**MITRE ATT&CK Mapping:** {technique_id}",
        f"**Severity Level:** {level.upper()}",
        f"**Rule Status:** {status}",
        "",
        "**False Positive Considerations:**",
    ])

    # Extract false positives from rule
    in_fp = False
    fp_found = False
    for line in rule_yaml.splitlines():
        stripped = line.strip()
        if stripped.startswith("falsepositives:"):
            in_fp = True
            continue
        if in_fp:
            if stripped.startswith("- "):
                parts.append(f"- {stripped[2:]}")
                fp_found = True
            elif stripped and not stripped.startswith("#"):
                break

    if not fp_found:
        parts.append("- Legitimate administrative activity matching the detection pattern")

    return "\n".join(parts)


# ============================================================================
# Query / Rule Builders
# ============================================================================

def _build_spl_query(technique_id: str, technique_name: str,
                      event: dict, sourcetype: str) -> str:
    """Build a SPL query from a log event."""
    lines = [f'index=main sourcetype="{sourcetype}"']

    if "process_name" in event:
        lines.append(f'  process_name="{event["process_name"]}"')
    if "action" in event:
        lines.append(f'  action="{event["action"]}"')
    if "event_id" in event:
        lines.append(f'  event_id={event["event_id"]}')
    if "command_line" in event:
        # Extract key pattern from command line
        cmd = event["command_line"]
        if "\\" in cmd or "/" in cmd:
            # Use wildcard pattern for paths
            key_part = cmd.split()[0] if " " in cmd else cmd[:30]
            lines.append(f'  command_line="*{key_part}*"')

    lines.append(f'| table _time, host, user, process_name, command_line, action, src_ip, dest_ip')
    lines.append(f'| sort -_time')
    lines.append(f'`comment("MITRE ATT&CK: {technique_id} - {technique_name}")`')

    return "\n".join(lines)


def _build_correlation_spl(technique_id: str, events: list[dict],
                            sourcetypes: list[str]) -> str:
    """Build a correlation SPL query across multiple sourcetypes."""
    st_filter = " OR ".join(f'sourcetype="{st}"' for st in sourcetypes)
    lines = [
        f"index=main ({st_filter})",
        f"| eval technique=\"{technique_id}\"",
        f"| stats count by host, sourcetype, action, _time span=5m",
        f"| stats dc(sourcetype) as source_count, values(sourcetype) as sources, "
        f"values(action) as actions by host",
        f"| where source_count >= {min(len(sourcetypes), 2)}",
        f"| table host, source_count, sources, actions",
    ]
    return "\n".join(lines)


def _build_kql_query(technique_id: str, technique_name: str,
                      event: dict) -> str:
    """Build a KQL query for Microsoft Sentinel."""
    # Determine appropriate table
    action = event.get("action", "")
    has_process = "process_name" in event or "command_line" in event

    if has_process:
        table = "DeviceProcessEvents"
        lines = [table, "| where TimeGenerated > ago(1h)"]
        if "process_name" in event:
            lines.append(f'| where FileName == "{event["process_name"]}"')
        if "command_line" in event:
            key_part = event["command_line"].split()[0] if " " in event["command_line"] else event["command_line"][:30]
            lines.append(f'| where ProcessCommandLine contains "{key_part}"')
        lines.extend([
            "| project TimeGenerated, DeviceName, AccountName, FileName, "
            "ProcessCommandLine, InitiatingProcessFileName",
            "| sort by TimeGenerated desc",
        ])
    else:
        table = "SecurityEvent"
        lines = [table, "| where TimeGenerated > ago(1h)"]
        if "action" in event:
            lines.append(f'| where Activity contains "{action}"')
        if "event_id" in event:
            lines.append(f'| where EventID == {event["event_id"]}')
        lines.extend([
            "| project TimeGenerated, Computer, Account, Activity, EventID",
            "| sort by TimeGenerated desc",
        ])

    return "\n".join(lines)


def _build_sigma_rule(technique_id: str, technique_name: str,
                       tactic: str, event: dict) -> str:
    """Build a Sigma detection rule YAML from a log event."""
    # Determine logsource category
    has_process = "process_name" in event
    has_network = "src_ip" in event and "dest_ip" in event

    if has_process:
        logsource = "    category: process_creation\n    product: windows"
    elif has_network:
        logsource = "    category: firewall\n    product: any"
    else:
        logsource = "    category: generic\n    product: any"

    # Build detection selection
    selection_fields = []
    if "process_name" in event:
        selection_fields.append(f"        {_sigma_field('Image', event['process_name'])}")
    if "parent_process_name" in event:
        selection_fields.append(f"        {_sigma_field('ParentImage', event['parent_process_name'])}")
    if "command_line" in event:
        cmd = event["command_line"]
        # Use contains modifier for command line matching
        key_parts = [p for p in cmd.split() if len(p) > 3][:3]
        if key_parts:
            selection_fields.append(f"        CommandLine|contains|all:")
            for part in key_parts:
                selection_fields.append(f"            - '{part}'")
    if "action" in event:
        selection_fields.append(f"        {_sigma_field('Action', event['action'])}")
    if "event_id" in event:
        selection_fields.append(f"        EventID: {event['event_id']}")

    if not selection_fields:
        selection_fields.append(f"        EventType: '*'")

    selection_block = "\n".join(selection_fields)
    safe_name = technique_name.replace("'", "").replace('"', "")
    safe_id = technique_id.lower().replace(".", "_")

    rule = f"""title: {safe_name} Detection via {technique_id}
id: {hashlib.md5(technique_id.encode()).hexdigest()[:8]}-{hashlib.md5(technique_name.encode()).hexdigest()[:4]}-{hashlib.md5(tactic.encode()).hexdigest()[:4]}-0000-{hashlib.md5(f"{technique_id}{technique_name}".encode()).hexdigest()[:12]}
status: experimental
description: Detects potential {safe_name} activity ({technique_id}) in the {tactic.replace('-', ' ')} phase.
references:
    - https://attack.mitre.org/techniques/{technique_id.replace('.', '/')}/
tags:
    - attack.{tactic}
    - attack.{technique_id.lower()}
logsource:
{logsource}
detection:
    selection:
{selection_block}
    condition: selection
falsepositives:
    - Legitimate administrative activity
    - Automated system maintenance
level: medium"""

    return rule


def _sigma_field(field_name: str, value: str) -> str:
    """Format a Sigma detection field."""
    if "*" in value or "?" in value:
        return f"{field_name}: '{value}'"
    return f"{field_name}: '{value}'"


def _yaml_field(yaml_text: str, field_name: str) -> str | None:
    """Extract a top-level YAML field value (simple string parsing)."""
    for line in yaml_text.splitlines():
        if line.startswith(f"{field_name}:"):
            val = line.split(":", 1)[1].strip()
            return val.strip("'\"") if val else None
    return None


# ============================================================================
# Analysis Helpers
# ============================================================================

def _compose_technique_analysis(technique_id: str, technique_name: str,
                                 tactic: str, event: dict) -> str:
    """Generate a brief technique analysis paragraph."""
    cmd = event.get("command_line", "")
    process = event.get("process_name", "")
    action = event.get("action", "")

    parts = [
        f"This event shows indicators of {technique_name} ({technique_id}), "
        f"which falls under the {tactic.replace('-', ' ').title()} tactic."
    ]

    if cmd:
        parts.append(f"The command line `{cmd[:150]}` is characteristic of this technique.")
    if process:
        parts.append(f"The process `{process}` is commonly leveraged in {technique_name} attacks.")
    if action:
        parts.append(f"The `{action}` action type confirms active execution of this technique.")

    return " ".join(parts)


def _compose_response_actions(technique_id: str, tactic: str,
                               severity: str) -> list[str]:
    """Generate response action recommendations."""
    actions = []

    # Universal actions
    actions.append("- Isolate the affected host if severity warrants immediate containment")
    actions.append("- Collect forensic artifacts (memory dump, disk image) for investigation")

    # Tactic-specific actions
    tactic_actions = {
        "persistence": "- Check for additional persistence mechanisms (registry, startup, services)",
        "execution": "- Review parent process chain for initial access vector",
        "credential-access": "- Force password reset for affected accounts",
        "lateral-movement": "- Audit recent authentication events across the network",
        "exfiltration": "- Review network traffic for data exfiltration indicators",
        "defense-evasion": "- Check for tampered security tool configurations",
        "privilege-escalation": "- Verify current privilege levels and revoke unauthorized access",
        "initial-access": "- Block the source IP/domain at the perimeter firewall",
        "discovery": "- Monitor for follow-up lateral movement or data access attempts",
        "collection": "- Check for staging directories and compressed archives",
        "command-and-control": "- Block identified C2 domains/IPs at DNS and firewall",
        "impact": "- Activate incident response plan and notify stakeholders",
    }
    if tactic in tactic_actions:
        actions.append(tactic_actions[tactic])

    # Severity-specific
    if severity.lower() in ("critical", "high"):
        actions.append("- Escalate to Tier 2/3 SOC analyst for immediate investigation")
    else:
        actions.append("- Add to investigation queue for detailed analysis")

    return actions


# ============================================================================
# Main Generator
# ============================================================================

def generate_training_data(simulator_url: str,
                           sigma_index: dict[str, str] | None = None,
                           output_path: Path | None = None,
                           limit: int = 0) -> list[dict]:
    """Generate all training examples from the simulator.

    Args:
        simulator_url: Base URL of the MDR Log Simulator.
        sigma_index: Optional {technique_id: rule_yaml} from SigmaHQ.
        output_path: Path to write JSONL output.
        limit: Max techniques to process (0 = all).

    Returns:
        List of training example dicts with {"messages": [...]}.
    """
    client = SimulatorClient(simulator_url)
    sigma_index = sigma_index or {}

    # Fetch all techniques
    techniques = client.get_techniques()
    if limit > 0:
        techniques = techniques[:limit]

    log.info(f"Processing {len(techniques)} techniques...")

    all_examples = []
    stats = {
        "log_analysis": 0,
        "siem_queries": 0,
        "sigma_rules": 0,
        "errors": 0,
    }

    for i, tech in enumerate(techniques):
        tech_id = tech["attack_id"]
        tech_name = tech["name"]
        tactics = tech.get("tactics", [])
        tactic = tactics[0] if tactics else "unknown"
        gen_count = tech.get("generator_count", 0)

        try:
            # Fetch preview logs
            preview = client.get_preview(tech_id)
            events = preview.get("events", [])
            sources = preview.get("generator_sources", [])

            if not events:
                log.debug(f"No events for {tech_id}, skipping.")
                continue

            # Fetch generator metadata for sourcetype info
            generators = client.get_generators(tech_id)
            gen_map = {g["generator_name"]: g for g in generators}

            examples = _generate_for_technique(
                tech_id, tech_name, tactic, events, sources,
                gen_map, client, sigma_index,
            )

            for ex in examples:
                all_examples.append(ex)
                # Count by domain based on question content
                q = ex["messages"][0]["content"].lower()
                if "sigma" in q:
                    stats["sigma_rules"] += 1
                elif "spl" in q or "kql" in q or "splunk" in q or "sentinel" in q or "correlation" in q:
                    stats["siem_queries"] += 1
                else:
                    stats["log_analysis"] += 1

        except (URLError, json.JSONDecodeError, KeyError) as e:
            log.warning(f"Error processing {tech_id}: {e}")
            stats["errors"] += 1
            continue

        if (i + 1) % 50 == 0:
            total = sum(v for k, v in stats.items() if k != "errors")
            log.info(f"  Processed {i+1}/{len(techniques)} techniques, "
                     f"{total} examples generated...")

        # Rate limit: ~150ms between techniques to stay under simulator limits
        time.sleep(0.15)

    # Write output
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for ex in all_examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        log.info(f"Wrote {len(all_examples)} examples to {output_path}")

    total = sum(v for k, v in stats.items() if k != "errors")
    log.info(f"\nGeneration complete:")
    log.info(f"  log_analysis:  {stats['log_analysis']:,}")
    log.info(f"  siem_queries:  {stats['siem_queries']:,}")
    log.info(f"  sigma_rules:   {stats['sigma_rules']:,}")
    log.info(f"  Total:         {total:,}")
    log.info(f"  Errors:        {stats['errors']:,}")

    return all_examples


def _generate_for_technique(tech_id: str, tech_name: str, tactic: str,
                             events: list[dict], sources: list[str],
                             gen_map: dict, client: SimulatorClient,
                             sigma_index: dict[str, str]) -> list[dict]:
    """Generate all training examples for a single technique."""
    examples = []

    # ── Log Analysis ──

    # 1. Single-source triage (one per unique source)
    seen_sources = set()
    for event, source in zip(events, sources):
        if source in seen_sources:
            continue
        seen_sources.add(source)

        sanitized = sanitize_log(event)
        source_name = generator_display_name(source)

        q = _select_template(SINGLE_SOURCE_TEMPLATES, f"{tech_id}_{source}").format(
            source_name=source_name,
            log=format_log_json(sanitized),
        )
        a = compose_single_source_answer(tech_id, tech_name, tactic, event, source_name)
        examples.append(_make_record(q, a))

    # 2. Multi-source correlation (if 2+ sources)
    if len(set(sources)) >= 2:
        unique_events = []
        unique_sources = []
        seen = set()
        for event, source in zip(events, sources):
            if source not in seen:
                seen.add(source)
                unique_events.append(event)
                unique_sources.append(source)

        source_names = [generator_display_name(s) for s in unique_sources]
        sanitized_events = [sanitize_log(e) for e in unique_events]

        # Build labeled logs block
        logs_parts = []
        for j, (sev, sname) in enumerate(zip(sanitized_events, source_names), 1):
            logs_parts.append(f"**Event {j} ({sname}):**\n```json\n{format_log_json(sev, compact=True)}\n```")
        logs_block = "\n\n".join(logs_parts)

        q = _select_template(MULTI_SOURCE_TEMPLATES, tech_id).format(
            source_count=len(unique_sources),
            logs_block=logs_block,
        )
        a = compose_multi_source_answer(tech_id, tech_name, tactic,
                                         unique_events, source_names)
        examples.append(_make_record(q, a))

    # 3. Benign/malicious triage (if we can get benign logs)
    if events and sources:
        primary_source = sources[0]
        # Map generator to log_type for benign generation
        log_type_map = {
            "endpoint_generator": "endpoint",
            "microsoft_defender": "endpoint",
            "crowdstrike": "endpoint",
            "palo_alto": "firewall",
            "fortinet": "firewall",
            "cisco_asa": "firewall",
            "okta": "authentication",
            "aws_cloudtrail": "cloud",
        }
        log_type = log_type_map.get(primary_source)
        if log_type:
            benign_events = client.get_benign_logs(log_type, count=4)
            if len(benign_events) >= 2:
                malicious_event = events[0]
                source_name = generator_display_name(primary_source)

                # Mix benign + malicious
                mixed = [sanitize_log(e) for e in benign_events[:4]]
                mal_sanitized = sanitize_log(malicious_event)
                # Insert malicious event at deterministic position
                insert_pos = int(hashlib.md5(tech_id.encode()).hexdigest(), 16) % (len(mixed) + 1)
                mixed.insert(insert_pos, mal_sanitized)

                logs_parts = []
                for j, evt in enumerate(mixed, 1):
                    logs_parts.append(f"**Event {j}:**\n```json\n{format_log_json(evt, compact=True)}\n```")
                logs_block = "\n\n".join(logs_parts)

                q = _select_template(TRIAGE_TEMPLATES, f"{tech_id}_triage").format(
                    event_count=len(mixed),
                    source_name=source_name,
                    logs_block=logs_block,
                )
                a = compose_triage_answer(tech_id, tech_name, tactic,
                                           malicious_event, insert_pos,
                                           len(mixed), source_name)
                examples.append(_make_record(q, a))

    # ── SIEM Queries ──

    if events:
        event = events[0]
        source = sources[0] if sources else "unknown"
        source_name = generator_display_name(source)
        sourcetype = event.get("sourcetype", "mdr:unknown")

        # SPL query
        q = _select_template(SPL_TEMPLATES, f"{tech_id}_spl").format(
            technique_id=tech_id,
            technique_name=tech_name,
            source_name=source_name,
            sourcetype=sourcetype,
        )
        a = compose_spl_answer(tech_id, tech_name, tactic, event,
                                sourcetype, source_name)
        examples.append(_make_record(q, a))

        # Multi-source correlation SPL (if 3+ sources)
        if len(set(sources)) >= 3:
            unique_events_spl = []
            unique_sources_spl = []
            unique_sts = []
            seen_spl = set()
            for ev, src in zip(events, sources):
                if src not in seen_spl:
                    seen_spl.add(src)
                    unique_events_spl.append(ev)
                    unique_sources_spl.append(src)
                    unique_sts.append(ev.get("sourcetype", "mdr:unknown"))

            source_list = ", ".join(generator_display_name(s) for s in unique_sources_spl)
            q = _select_template(CORRELATION_SPL_TEMPLATES, f"{tech_id}_corr").format(
                technique_id=tech_id,
                technique_name=tech_name,
                source_list=source_list,
            )
            a = compose_correlation_spl_answer(tech_id, tech_name, tactic,
                                                unique_events_spl, unique_sts,
                                                [generator_display_name(s) for s in unique_sources_spl])
            examples.append(_make_record(q, a))

        # KQL query (for techniques with Defender/Azure sources or all with process data)
        azure_sources = {"microsoft_defender", "azure_activity", "office365"}
        has_azure = any(s in azure_sources for s in sources)
        has_process = "process_name" in event or "command_line" in event
        if has_azure or has_process:
            q = _select_template(KQL_TEMPLATES, f"{tech_id}_kql").format(
                technique_id=tech_id,
                technique_name=tech_name,
            )
            a = compose_kql_answer(tech_id, tech_name, tactic, event, source_name)
            examples.append(_make_record(q, a))

    # ── Sigma Rules ──

    if events:
        event = events[0]
        sanitized = sanitize_log(event)

        # Write Sigma rule
        q = _select_template(SIGMA_WRITE_TEMPLATES, f"{tech_id}_sigma").format(
            technique_id=tech_id,
            technique_name=tech_name,
            log=format_log_json(sanitized),
        )
        a = compose_sigma_write_answer(tech_id, tech_name, tactic, event)
        examples.append(_make_record(q, a))

        # Explain existing SigmaHQ rule (if available)
        if tech_id in sigma_index:
            rule_yaml = sigma_index[tech_id]
            q = _select_template(SIGMA_EXPLAIN_TEMPLATES, f"{tech_id}_explain").format(
                rule=rule_yaml,
            )
            a = compose_sigma_explain_answer(rule_yaml, tech_id)
            examples.append(_make_record(q, a))

    return examples


def _make_record(user_msg: str, assistant_msg: str) -> dict:
    """Create a standard training record."""
    return {
        "messages": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate attack log training data from MDR Log Simulator"
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="Output JSONL file path")
    parser.add_argument("--simulator-url", default="http://192.168.1.67:6971",
                        help="MDR Log Simulator base URL")
    parser.add_argument("--with-sigma-hq", action="store_true",
                        help="Download and include SigmaHQ rules for explain examples")
    parser.add_argument("--sigma-cache", type=Path, default=None,
                        help="SigmaHQ cache directory")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max techniques to process (0 = all)")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("  Attack Log Training Data Generator")
    log.info("=" * 60)
    log.info(f"  Simulator: {args.simulator_url}")
    log.info(f"  Output:    {args.output}")
    log.info(f"  SigmaHQ:   {'enabled' if args.with_sigma_hq else 'disabled'}")
    if args.limit:
        log.info(f"  Limit:     {args.limit} techniques")
    log.info("")

    # Build SigmaHQ index if requested
    sigma_index = {}
    if args.with_sigma_hq:
        from scripts.sources.sigma_hq import build_sigma_index
        sigma_index = build_sigma_index(cache_dir=args.sigma_cache)
        log.info(f"SigmaHQ: {len(sigma_index)} technique rules loaded.\n")

    # Generate training data
    examples = generate_training_data(
        simulator_url=args.simulator_url,
        sigma_index=sigma_index,
        output_path=args.output,
        limit=args.limit,
    )

    log.info(f"\nDone. {len(examples)} total examples written to {args.output}")


if __name__ == "__main__":
    main()
