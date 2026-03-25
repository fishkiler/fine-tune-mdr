# AI-Powered Cybersecurity Threat Detection

## Fine-Tuning Plan for DGX Spark

*MITRE ATT&CK + CVE + ExploitDB | Local, Offline, No Cloud Dependencies*

---

| | |
|---|---|
| **Base Model** | Cisco Foundation-Sec-8B (Llama 3.1 8B, pre-trained on 5.1B cybersecurity tokens) |
| **Dataset** | jason-oneal/mitre-stix-cve-exploitdb-dataset (~245K examples, Apache-2.0) |
| **Method** | QLoRA fine-tuning via Unsloth (2x speed, 70% less VRAM) |
| **Hardware** | DGX Spark (Grace Blackwell, 128 GB unified memory, ~1 PFLOP FP4) |
| **Est. Time** | 4–12 hours fine-tuning | 1–2 weeks total prototype |

---

## 1. Base Model: Why Foundation-Sec-8B

Rather than starting from a generic Llama 3.1 8B, the refined plan uses Cisco's Foundation-Sec-8B as the base. This model was purpose-built for cybersecurity and already understands the domain before you even start fine-tuning.

### Key Advantages

- Pre-trained on 5.1 billion tokens of cybersecurity-specific data (CVEs, threat reports, ATT&CK content, security documentation)
- Outperforms Llama 3.1 8B on all cybersecurity benchmarks and matches or exceeds Llama 3.1 70B on several
- Same Llama 3.1 8B architecture — fully compatible with all existing tooling (Unsloth, PEFT, transformers)
- Open-weight, permissive license — run on-prem, air-gapped, no restrictions
- Specifically designed for SOC acceleration, TTP mapping, and vulnerability analysis — exactly your use case

*By fine-tuning a model that already speaks cybersecurity fluently, you get dramatically better results with less data and fewer training epochs than starting from a general-purpose base.*

---

## 2. Dataset Strategy

### Primary Dataset

The jason-oneal/mitre-stix-cve-exploitdb-dataset-alpaca-chatml-harmony dataset on Hugging Face provides the core training data. It contains approximately 245,000 validated examples covering:

- CVE vulnerability descriptions with CVSS severity scores from MITRE and NVD
- Exploit code references and proof-of-concept snippets from ExploitDB
- MITRE ATT&CK reasoning Q&A pairs (tactics, techniques, procedures)
- TTP mapping scenarios from real-world threat intelligence reports
- Secure coding dialogues and vulnerability remediation guidance

### Data Formats Available

The dataset ships in three formats. Use ChatML for fine-tuning since it maps naturally to the instruction-following format Foundation-Sec-8B expects:

| Format | Structure | Best For |
|--------|-----------|----------|
| **ChatML** | `messages: [{role, content}, ...]` | Instruction fine-tuning **(recommended)** |
| **Alpaca** | `instruction / input / output` fields | Simple Q&A fine-tuning |
| **Harmony** | Raw tokenized text strings | Continued pre-training |

### Building Fresh Data with pentestds

The dataset is built by the open-source pentestds pipeline (github.com/jason-allen-oneal/pentest-dataset-builder). You can run this pipeline yourself to generate fresh data with the latest CVEs, exploits, and ATT&CK updates. First run takes 10–20 minutes; subsequent refreshes take 2–5 minutes.

### Supplemental Data (Optional)

- Your own labeled log data (syslog, firewall, EDR feeds) — the more domain-specific, the better
- cobo512/Mitre-ATTACK-reasoning-dataset — 1,600 Q&A pairs on ATT&CK tactics/techniques
- Zeek/Suricata public log datasets — real network traffic with labeled attacks
- Public CTF writeups and red-team reports — helps the model learn real-world breach narratives

---

## 3. Environment Setup

### Step 1: Build the Unsloth Docker Container

NVIDIA and Unsloth provide an official Docker image optimized for DGX Spark. This gives you a clean, reproducible environment with all dependencies pre-configured.

```bash
sudo apt update && sudo apt install -y wget
wget -O Dockerfile "https://raw.githubusercontent.com/unslothai/notebooks/main/Dockerfile_DGX_Spark"
docker build -t unsloth-dgx-spark .
docker run -it --gpus=all --net=host --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $(pwd):$(pwd) -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -w $(pwd) unsloth-dgx-spark
```

### Step 2: Build the Dataset Pipeline

```bash
git clone https://github.com/jason-allen-oneal/pentest-dataset-builder.git
cd pentest-dataset-builder
python -m venv .venv && source .venv/bin/activate
pip install -U pip && pip install -r requirements.txt && pip install -e .
export NVD_API_KEY="your-key-here"   # optional, prevents throttling
pentestds build
```

Output files land in `data/datasets/dist/` — `alpaca_train.jsonl`, `chatml_train.jsonl`, `harmony_train.jsonl`, plus validation splits and `provenance.json`.

### Step 3: Pre-Flight Memory Check

**IMPORTANT:** Before starting any fine-tuning run, ensure the DGX Spark's 128 GB unified memory is free. The training process needs memory for the model weights, LoRA adapters, optimizer states (AdamW keeps two copies of gradients), data batches, gradient accumulation buffers, and activation caches. A full QLoRA run on an 8B model can consume 20–40 GB depending on sequence length and batch size. Any other models or services running in the background compete for the same memory pool and will cause slower training or out-of-memory crashes mid-run.

**Before every training session, run these checks:**

- Stop Ollama: `ollama stop` (or kill the process entirely)
- Stop any inference servers: vLLM, LM Studio, text-generation-webui, or FastAPI wrappers serving other models
- Close any other GPU-heavy applications
- Verify memory is clear by running: `nvidia-smi` or `tegrastats`
- Confirm that GPU memory usage is near zero before launching the training script

Once training is complete and you switch to inference-only mode, memory pressure drops significantly and you can safely run other services alongside the model.

---

## 4. Fine-Tuning Configuration

### Recommended Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Method | QLoRA (4-bit) | 4x less VRAM than full LoRA |
| LoRA Rank (r) | 16–32 | Higher = more capacity, more memory |
| LoRA Alpha | r × 2 (e.g., 32–64) | Scales adapter strength |
| Learning Rate | 2e-4 | Unsloth default, good starting point |
| Epochs | 1–3 | More than 3 risks overfitting |
| Batch Size | 4–8 (per device) | Adjust based on memory headroom |
| Gradient Accumulation | 4 | Effective batch = batch × accumulation |
| Max Sequence Length | 512–1024 | 512 for most ATT&CK Q&A, 1024 for logs |
| Optimizer | AdamW 8-bit | Memory-efficient, standard for LoRA |
| Target Modules | q_proj, k_proj, v_proj, o_proj | Attention layers (Unsloth default) |

### Estimated Training Time on DGX Spark

With the 245K-example dataset and the above configuration: approximately 4–12 hours depending on sequence length and batch size. The 128 GB unified memory means the 8B model in 4-bit quantization uses roughly 5–8 GB, leaving ample headroom for data batching and optimizer states.

---

## 5. Training Script Outline

The following is a high-level script structure using Unsloth + Hugging Face transformers. Adapt paths and hyperparameters to your specific setup.

```python
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# 1. Load Foundation-Sec-8B with QLoRA
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="fdtn-ai/Foundation-Sec-8B",
    max_seq_length=1024, dtype=None, load_in_4bit=True)

# 2. Add LoRA adapters
model = FastLanguageModel.get_peft_model(model, r=16, lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_dropout=0, bias="none", use_gradient_checkpointing="unsloth")

# 3. Load dataset (ChatML format)
dataset = load_dataset("jason-oneal/mitre-stix-cve-exploitdb-dataset-....",
    data_files={"train": "chatml_train.jsonl"})

# 4. Train
trainer = SFTTrainer(model=model, tokenizer=tokenizer,
    train_dataset=dataset["train"],
    args=TrainingArguments(output_dir="./outputs", per_device_train_batch_size=4,
        gradient_accumulation_steps=4, num_train_epochs=2,
        learning_rate=2e-4, save_strategy="epoch", logging_steps=25))
trainer.train()

# 5. Save & export
model.save_pretrained("threat-detector-v1")
model.save_pretrained_gguf("threat-detector-v1-gguf", tokenizer)  # for Ollama
```

---

## 6. Automated Inference Pipeline

Once the model is trained, deploy it as an automated threat detection service running entirely on the DGX Spark.

### Architecture

1. **Log Collector:** Cron job pulls fresh logs every 5 minutes (syslog, firewall, EDR feeds).
2. **Parser:** Chunks logs by time window and event type, formats them as model prompts.
3. **Inference Engine:** FastAPI/Flask wraps the model. Each chunk is classified against ATT&CK techniques.
4. **Confidence Filter:** Results below 70% confidence are flagged for human review.
5. **Alerting:** High-confidence detections fire alerts to Slack, email, or SIEM dashboard.
6. **Queue:** Redis or similar ensures log spikes don't crash the pipeline.

### Prompt Template for Log Analysis

```
Given the following log entry, identify the most likely MITRE ATT&CK
technique being used. Provide the technique ID, tactic, a confidence
score (0-100), and a brief explanation.

Log: {log_entry}

Response format: T[ID] | [Tactic] | [Confidence]% | [Explanation]
```

---

## 7. Ongoing Maintenance

### Monthly Incremental Fine-Tuning

You do not retrain from scratch. Use incremental fine-tuning to evolve the model:

1. Run `pentestds build` to pull latest CVEs, exploits, and ATT&CK updates.
2. Add any new labeled logs from your environment.
3. Load the last saved checkpoint and run a quick LoRA pass (1–2 hours).
4. Validate on held-out logs before merging new weights into production.
5. If false-positive rate exceeds 10%, trigger an unscheduled retrain.

### Trigger-Based Retraining

Beyond the monthly schedule, retrain when: a major new ATT&CK version drops (typically twice per year), a significant zero-day campaign hits your sector, or the model's false-positive rate drifts above your threshold.

---

## 8. Deployment Portability

The fine-tuned model is not locked to the DGX Spark. It exports as standard Hugging Face safetensors and can run anywhere:

| Target | How |
|--------|-----|
| Any NVIDIA GPU (≥24 GB) | Load with transformers or vLLM directly |
| Consumer GPUs (RTX 4090+) | Export to GGUF / AWQ / GPTQ quantized formats |
| CPU-only servers | Run via llama.cpp or Ollama (slower but functional) |
| Cloud (AWS/GCP/Azure) | Deploy container image to any cloud GPU instance |
| Edge devices | Jetson Orin, laptops with RTX cards |

The DGX Spark is optimized for training and development. Inference can happen anywhere — no vendor lock-in.

---

## 9. Implementation Timeline

| Phase | Tasks | Duration |
|-------|-------|----------|
| **Day 1** | Set up Unsloth Docker on Spark, clone pentestds, build dataset, obtain NVD API key | 2–4 hours |
| **Day 2–3** | Download Foundation-Sec-8B, configure QLoRA, run first training pass on ChatML data | 4–12 hours training |
| **Day 4–5** | Evaluate on held-out data, tune hyperparameters, run second pass if needed | 4–8 hours |
| **Day 6–7** | Build FastAPI inference wrapper, set up log parsing, test on sample logs | 1–2 days |
| **Week 2** | Integrate with real log feeds, configure alerting (Slack/email), set up cron automation | 3–5 days |
| **Ongoing** | Monthly incremental fine-tuning, monitor false-positive rate, refresh dataset | 1–2 hours/month |

---

## 10. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Overfitting on structured ATT&CK data | Model gives textbook answers, misses novel attacks | Mix in real-world logs, breach reports, CTF data; keep epochs ≤3 |
| Hallucinated threat IDs | False alerts overwhelm SOC team | Confidence threshold at 70%; human review queue for low-confidence |
| Model drift over time | Accuracy degrades as threats evolve | Monthly incremental fine-tuning; automated false-positive monitoring |
| Dataset quality issues | Garbage in, garbage out | Dataset is 98% validated via Pydantic schemas; provenance tracking on every record |

---

## 11. Key Resources

- **Base Model:** huggingface.co/fdtn-ai/Foundation-Sec-8B
- **Training Dataset:** huggingface.co/datasets/jason-oneal/mitre-stix-cve-exploitdb-dataset-alpaca-chatml-harmony
- **Dataset Pipeline:** github.com/jason-allen-oneal/pentest-dataset-builder
- **Unsloth on DGX Spark:** docs.unsloth.ai/basics/fine-tuning-llms-with-nvidia-dgx-spark-and-unsloth
- **NVIDIA DGX Spark Playbooks:** github.com/NVIDIA/dgx-spark-playbooks
- **NVD API Key (for pipeline):** nvd.nist.gov/developers/request-an-api-key
- **Foundation-Sec-8B Technical Report:** arxiv.org/abs/2504.21039

---

> **Ready to build. Clone, train, deploy — all on your desk, no cloud required.**
