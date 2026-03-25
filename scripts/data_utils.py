"""
============================================================================
Fine-Tune MDR — Shared Data Utilities
============================================================================
Parsing functions for custom training data (markdown + JSONL).
Used by both prepare_data.py and refresh_data.py.
============================================================================
"""

import json
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a cybersecurity threat analyst. Analyze the following log entry "
    "and identify potential MITRE ATT&CK techniques. Respond in this exact "
    "format: T[ID] | [Tactic] | [Confidence]% | [Explanation]"
)


# ============================================================================
# Markdown Parser
# ============================================================================

def parse_markdown_examples(filepath: str) -> list[list[dict]]:
    """
    Parse a markdown file into a list of conversations.

    Format:
        ---
        **system**: (optional) Custom system prompt
        **user**: User message
        **assistant**: Assistant response
        ---

    Returns list of conversations, where each conversation is a list of
    {"role": ..., "content": ...} dicts.
    """
    text = Path(filepath).read_text(encoding="utf-8")

    # Split on --- separators
    blocks = re.split(r"\n---\s*\n", text)

    conversations = []
    turn_pattern = re.compile(
        r"\*\*(\w+)\*\*:\s*(.*?)(?=\n\*\*\w+\*\*:|\Z)",
        re.DOTALL,
    )

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Skip comment-only blocks (lines starting with #)
        non_comment_lines = [
            line for line in block.split("\n")
            if line.strip() and not line.strip().startswith("#")
        ]
        if not non_comment_lines:
            continue

        turns = turn_pattern.findall(block)
        if not turns:
            continue

        messages = []
        has_system = False
        for role, content in turns:
            role = role.lower().strip()
            content = content.strip()
            if not content:
                continue
            if role == "system":
                has_system = True
            messages.append({"role": role, "content": content})

        if not messages:
            continue

        # Inject default system prompt if none provided
        if not has_system:
            messages.insert(0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT})

        conversations.append(messages)

    return conversations


# ============================================================================
# JSONL Parser
# ============================================================================

def parse_jsonl_examples(filepath: str) -> list[list[dict]]:
    """
    Parse a JSONL file into a list of conversations.

    Each line should be: {"messages": [{"role": "...", "content": "..."}, ...]}
    """
    conversations = []
    with open(filepath, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                log.warning(f"{filepath}:{line_num} — invalid JSON: {e}")
                continue

            # Support both {"messages": [...]} and bare [...]
            if isinstance(data, list):
                messages = data
            elif isinstance(data, dict) and "messages" in data:
                messages = data["messages"]
            else:
                log.warning(f"{filepath}:{line_num} — expected 'messages' key or array")
                continue

            # Inject default system prompt if none
            if not any(m.get("role") == "system" for m in messages):
                messages.insert(0, {"role": "system", "content": DEFAULT_SYSTEM_PROMPT})

            conversations.append(messages)

    return conversations


# ============================================================================
# Custom Data Loader
# ============================================================================

def load_custom_data(tokenizer, project_root: str = ".") -> list[str]:
    """
    Load all custom training examples from:
      - {project_root}/data/custom/*.md
      - {project_root}/data/custom/*.jsonl

    Returns list of formatted text strings ready for training.
    """
    root = Path(project_root)
    all_conversations: list[list[dict]] = []

    # 1. data/custom/*.md
    custom_dir = root / "data" / "custom"
    if custom_dir.exists():
        for md_file in sorted(custom_dir.glob("*.md")):
            if md_file.name == "README.md":
                continue
            convos = parse_markdown_examples(str(md_file))
            log.info(f"Loaded {len(convos)} examples from {md_file.name}")
            all_conversations.extend(convos)

        # 2. data/custom/*.jsonl
        for jsonl_file in sorted(custom_dir.glob("*.jsonl")):
            convos = parse_jsonl_examples(str(jsonl_file))
            log.info(f"Loaded {len(convos)} examples from {jsonl_file.name}")
            all_conversations.extend(convos)

    if not all_conversations:
        log.info("No custom examples found.")
        return []

    # Apply chat template to all conversations
    log.info(f"Formatting {len(all_conversations)} custom examples with chat template...")
    formatted_texts = []
    for i, messages in enumerate(all_conversations):
        try:
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            formatted_texts.append(text)
        except Exception as e:
            log.warning(f"Failed to format custom example {i + 1}: {e}")

    log.info(f"Formatted {len(formatted_texts)} custom examples.")
    return formatted_texts


def load_custom_conversations(project_root: str = ".") -> list[list[dict]]:
    """
    Load all custom training examples as raw conversations (no tokenizer needed).

    Returns list of conversations, where each conversation is a list of
    {"role": ..., "content": ...} dicts.
    """
    root = Path(project_root)
    all_conversations: list[list[dict]] = []

    custom_dir = root / "data" / "custom"
    if custom_dir.exists():
        for md_file in sorted(custom_dir.glob("*.md")):
            if md_file.name == "README.md":
                continue
            convos = parse_markdown_examples(str(md_file))
            log.info(f"Loaded {len(convos)} examples from {md_file.name}")
            all_conversations.extend(convos)

        for jsonl_file in sorted(custom_dir.glob("*.jsonl")):
            convos = parse_jsonl_examples(str(jsonl_file))
            log.info(f"Loaded {len(convos)} examples from {jsonl_file.name}")
            all_conversations.extend(convos)

    return all_conversations
