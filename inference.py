#!/usr/bin/env python3
"""
============================================================================
Fine-Tune MDR — Production Inference Server
============================================================================
FastAPI server for cybersecurity threat analysis using the fine-tuned model.
Returns structured threat assessments with calibrated confidence scores
and tiered alerting.

Usage:
    uvicorn inference:app --host 0.0.0.0 --port 8080
    python inference.py  # runs uvicorn directly
============================================================================
"""

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ============================================================================
# Config
# ============================================================================

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


CFG = load_config()

# ============================================================================
# State
# ============================================================================

model = None
tokenizer = None
start_time = None
request_count = 0


# ============================================================================
# Lifespan
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, start_time

    log.info("Loading fine-tuned model...")
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import get_chat_template

    model_dir = CFG["training"]["output_dir"]
    model_cfg = CFG["model"]

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        max_seq_length=model_cfg["max_seq_length"],
        dtype=model_cfg["dtype"],
        load_in_4bit=True,
    )
    tokenizer = get_chat_template(tokenizer, chat_template=model_cfg["chat_template"])
    FastLanguageModel.for_inference(model)

    start_time = time.time()
    log.info(f"Model loaded from {model_dir}. Server ready.")

    yield

    log.info("Shutting down inference server.")


app = FastAPI(title="Fine-Tune MDR Inference", lifespan=lifespan)


# ============================================================================
# Schemas
# ============================================================================

class AnalyzeRequest(BaseModel):
    log_entry: str
    system_prompt: str | None = None


class ThreatAssessment(BaseModel):
    technique_id: str | None
    tactic: str | None
    confidence: float
    calibrated_confidence: float
    explanation: str | None
    alert_tier: str
    raw_response: str


class HealthResponse(BaseModel):
    status: str
    uptime_seconds: float
    request_count: int
    model_dir: str
    timestamp: str


# ============================================================================
# Confidence Extraction
# ============================================================================

def extract_confidence_from_logprobs(scores: list[torch.Tensor], temperature: float) -> float:
    """
    Extract calibrated confidence from token log-probabilities.

    Uses the mean of max token probabilities across generated tokens,
    then applies temperature scaling.
    """
    if not scores:
        return 0.5

    max_probs = []
    for score in scores:
        probs = torch.softmax(score.squeeze(0), dim=-1)
        max_probs.append(probs.max().item())

    raw_confidence = float(np.mean(max_probs))

    # Apply temperature scaling (in logit space)
    eps = 1e-7
    logit = np.log(np.clip(raw_confidence, eps, 1 - eps) / np.clip(1 - raw_confidence, eps, 1 - eps))
    calibrated = 1 / (1 + np.exp(-logit / max(temperature, 0.01)))

    return float(calibrated)


def get_alert_tier(confidence: float) -> str:
    """Map calibrated confidence to alert tier."""
    tiers = CFG["inference"]["alert_tiers"]
    if confidence >= tiers["auto_alert"]:
        return "auto_alert"
    elif confidence >= tiers["needs_verification"]:
        return "needs_verification"
    elif confidence >= tiers["human_review"]:
        return "human_review"
    else:
        return "log_only"


# ============================================================================
# Response Parsing
# ============================================================================

import re

RESPONSE_PATTERN = re.compile(
    r"T(\d{4}(?:\.\d{3})?)\s*\|\s*(\w[\w\s]*?)\s*\|\s*(\d+)%?\s*\|\s*(.+)",
    re.IGNORECASE,
)


def parse_response(text: str) -> dict:
    """Parse structured model output."""
    for line in text.strip().split("\n"):
        m = RESPONSE_PATTERN.search(line)
        if m:
            return {
                "technique_id": f"T{m.group(1)}",
                "tactic": m.group(2).strip(),
                "self_reported_confidence": int(m.group(3)),
                "explanation": m.group(4).strip(),
            }
    return {"technique_id": None, "tactic": None, "self_reported_confidence": None, "explanation": None}


# ============================================================================
# Routes
# ============================================================================

@app.post("/analyze", response_model=ThreatAssessment)
async def analyze_log(request: AnalyzeRequest):
    """Analyze a log entry for cybersecurity threats."""
    global request_count
    request_count += 1

    system_prompt = request.system_prompt or (
        "You are a cybersecurity threat analyst. Analyze the following log entry and "
        "identify potential MITRE ATT&CK techniques. Respond in this exact format:\n"
        "T[ID] | [Tactic] | [Confidence]% | [Explanation]"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": request.log_entry},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=CFG["model"]["max_seq_length"],
    ).to(model.device)

    inf_cfg = CFG["inference"]
    temperature = CFG["calibration"]["temperature"]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=inf_cfg["max_new_tokens"],
            do_sample=False,  # greedy for determinism
            return_dict_in_generate=True,
            output_scores=True,
        )

    # Decode response
    new_tokens = outputs.sequences[0][inputs["input_ids"].shape[1]:]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    # Extract confidence from actual token probabilities
    calibrated_conf = extract_confidence_from_logprobs(outputs.scores, temperature)

    # Parse structured output
    parsed = parse_response(response_text)

    return ThreatAssessment(
        technique_id=parsed["technique_id"],
        tactic=parsed["tactic"],
        confidence=calibrated_conf,
        calibrated_confidence=calibrated_conf,
        explanation=parsed["explanation"],
        alert_tier=get_alert_tier(calibrated_conf),
        raw_response=response_text,
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Server health status."""
    uptime = time.time() - start_time if start_time else 0
    return HealthResponse(
        status="healthy",
        uptime_seconds=round(uptime, 1),
        request_count=request_count,
        model_dir=CFG["training"]["output_dir"],
        timestamp=datetime.now().isoformat(),
    )


# ============================================================================
# Direct Run
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "inference:app",
        host=CFG["inference"]["host"],
        port=CFG["inference"]["port"],
        log_level="info",
    )
