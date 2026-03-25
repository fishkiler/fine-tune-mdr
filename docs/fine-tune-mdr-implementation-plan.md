# fine-tune-mdr — Implementation Plan
## Based on repo analysis: github.com/fishkiler/fine-tune-mdr

*Generated: March 25, 2026*

---

## What This Repo Actually Is

After reading `dgx-spark-fine-tuning-plan.md`, `config.yaml`, and `defender.py`, the picture is clear. This is **not** a simple fine-tuning script. It's a multi-track AI training platform with three distinct training paradigms running in parallel:

| Track | File(s) | Model | Purpose |
|---|---|---|---|
| **SFT (supervised)** | `train.py`, `train.sh` | Foundation-Sec-8B-Instruct | Core cybersecurity fine-tune on CVE/ATT&CK data |
| **Game VLM adapter** | `train_game_adapter.py`, `train_game.sh` | Qwen3.5-9B | Multimodal game-playing (Pac-Man) via vision |
| **RL (PPO)** | `defender.py`, `train_native.py` | PPO CNN policy | Atari Defender as RL training environment |

The Colab migration is complete. The DGX Spark memory config (`90-dgx-spark-memory.conf`) is already committed. The dashboard streams live to port 28000. The attack log source (`http://192.168.1.67:6971`) is already wired in `config.yaml`.

**This plan does not rebuild what exists. It extends it.**

---

## Critical Issues Found in config.yaml

### Issue 1 — Flash Attention misconfigured for DGX Spark

```yaml
# Current (WRONG for DGX Spark):
attn_implementation: "flash_attention_2"  # comment says "use sdpa on DGX GB10"

# The comment is correct but the value is wrong.
# sdpa works but is slow. The right answer is to BUILD flash-attn targeting sm_120:
```

**Fix (one-time, ~75 min compile):**
```bash
export TORCH_CUDA_ARCH_LIST="12.0"
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
export CUDA_HOME=/usr/local/cuda
python setup.py install
```

Then update `config.yaml`:
```yaml
attn_implementation: "flash_attention_2"   # now correct — sm_120 build runs on sm_121
```

And for the game adapter section:
```yaml
game_adapters:
  attention: "flash_attention_2"           # upgrade from "sdpa"
```

**Impact:** Foundation-Sec-8B training throughput improves significantly. Qwen3.5-9B game adapter training also benefits. This is the single highest-leverage change in this document.

### Issue 2 — CUDA runtime path not set

The DGX Spark ships CUDA 13.0 but vLLM and some training libs link against CUDA 12. Add to your `.env` or shell profile:

```bash
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH}"
export VLLM_MOE_KERNEL_BACKEND=triton
export VLLM_DISABLED_KERNELS=cutlass_moe_mm,cutlass_scaled_mm
export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas
```

### Issue 3 — `torch._scaled_mm` padding bug on sm_121

If you run anything that calls `w8a8_utils.py` in vLLM (W8A8 quantized inference), you'll hit a shape mismatch. Patch:

```python
# In vllm/model_executor/layers/quantization/utils/w8a8_utils.py
if output.numel() > expected_numel:
    output = output[:expected_numel].reshape(expected_shape)
```

---

## Phase 0 — Prerequisite: nanoGPT Shakespeare (1 afternoon)

**Why do this first:** Before touching Foundation-Sec-8B or Qwen3.5-9B, spend one afternoon with a model small enough that every hyperparameter change shows a visible effect within 3 minutes. This builds intuition for the dials you'll be turning in Phases 1–3.

**Setup:**
```bash
git clone https://github.com/karpathy/nanoGPT
cd nanoGPT
python data/shakespeare_char/prepare.py
python train.py config/train_shakespeare_char.py
```

**Experiments to run (in order):**

| Experiment | Change | Expected Effect |
|---|---|---|
| Baseline | defaults | ~1.47 val loss |
| LR too high | `learning_rate = 1e-2` | Loss explodes |
| LR too low | `learning_rate = 1e-5` | Loss barely moves |
| No dropout | `dropout = 0.0` | Overfits faster |
| Half depth | `n_layer = 3` (from 6) | Underfits, faster |
| Double heads | `n_head = 8` (from 4) | Marginal improvement |
| Tiny batch | `batch_size = 16` | Noisier loss curve |

**Goal:** When you can predict what a hyperparameter change will do before you run it, move to Phase 1.

---

## Phase 1 — autoresearch Overnight Loop (1–2 nights)

**What this adds:** Autonomous experiment iteration on the DGX Spark using Karpathy's `autoresearch` framework. You sleep, it runs ~100 experiments, you read the log in the morning.

**Setup:**
```bash
git clone https://github.com/karpathy/autoresearch
cd autoresearch

# Apply Flash Attention fix first
export TORCH_CUDA_ARCH_LIST="12.0"

# Install
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
uv run prepare.py
```

**DGX Spark-specific `program.md` additions:**

Add these notes to the autoresearch `program.md` for the agent:
```markdown
## Hardware context
- DGX Spark: NVIDIA GB10 Superchip, sm_121, 128GB unified memory
- Flash Attention compiled for sm_120 (runs natively on sm_121 via forward compatibility)
- CUDA 13.0 — do NOT use torch_compile (causes recompilation spikes on Blackwell)
- SDPA available as fallback: attn_implementation="sdpa"
- Muon optimizer works well on this hardware
- BF16 is faster than FP16 on Blackwell

## What to explore
- Architecture changes (depth, heads, FFN multiplier)
- Optimizer settings (Muon momentum, AdamW beta2)
- Banded vs full attention patterns (WINDOW_PATTERN: "L" vs "SSSL")
- Sequence length vs batch size tradeoffs
```

**What to read in the morning:** The agent's experiment log shows val_bpb (validation bits per byte) per run. Look for: which architecture changes consistently help, which hyperparameter ranges the agent gravitates toward, and whether the agent found anything surprising (e.g. smaller depth + wider FFN outperforming the baseline).

This phase teaches you what autoresearch found — and that knowledge directly informs how you tune Foundation-Sec-8B in Phase 2.

---

## Phase 2 — Foundation-Sec-8B SFT (existing track, enhanced)

**Status:** Already designed and partially implemented. `train.py`, `config.yaml`, and `dgx-spark-fine-tuning-plan.md` are all in place.

**Changes to make before running:**

### 2a. Fix Flash Attention (see Issue 1 above)

### 2b. Add mdr-log-simulator as a live data source

Your `config.yaml` already has this configured:
```yaml
sources:
  attack_logs:
    enabled: true
    simulator_url: "http://192.168.1.67:6971"
    sigma_hq: true
```

The simulator generates CEF logs across 78 sources from 24 MDR providers. Add an export script to convert simulator output to ChatML format for training:

```python
# scripts/export_simulator_data.py
import httpx, json
from pathlib import Path

def export_to_chatml(simulator_url: str, output_path: str, n_samples: int = 5000):
    """Pull events from mdr-log-simulator and format as ChatML training pairs."""
    records = []
    # Pull events from simulator
    resp = httpx.get(f"{simulator_url}/events", params={"limit": n_samples})
    events = resp.json()
    
    for event in events:
        record = {
            "messages": [
                {
                    "role": "user",
                    "content": f"Analyze this security log event and identify the MITRE ATT&CK technique:\n\n{event['raw']}"
                },
                {
                    "role": "assistant", 
                    "content": f"{event['technique_id']} | {event['tactic']} | {event['confidence']}% | {event['explanation']}"
                }
            ]
        }
        records.append(record)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
    
    print(f"Exported {len(records)} records to {output_path}")
```

### 2c. Domain weight tuning (from config.yaml)

The current export weights in `config.yaml` are well-reasoned. Worth noting the rationale:

```yaml
domain_weights:
  cve: 0.08              # 618K records — downsample heavily, most are repetitive
  mitre_attack: 10.0     # 1,317 records — oversample 10x, high signal
  secure_code_review: 30.0  # 348 records — extremely high signal, oversample aggressively
  siem_queries: 8.0      # ~700 records — SPL/KQL syntax, critical for Splunk integration
  sigma_rules: 10.0      # ~700 records — YAML rules, must be syntactically perfect
  log_analysis: 5.0      # ~1600 records — direct MDR relevance
```

The `siem_queries` and `sigma_rules` weights are especially important for your Splunk ES / Crucible use case. Consider bumping `siem_queries` to 12.0 if Splunk SPL generation is a primary goal.

### 2d. Training run checklist

Before every training session:
```bash
# 1. Stop inference servers
sudo systemctl stop ollama 2>/dev/null || true
pkill -f vllm 2>/dev/null || true

# 2. Set environment
export TORCH_CUDA_ARCH_LIST="12.0"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH}"
export PYTORCH_ALLOC_CONF=expandable_segments:True

# 3. Verify memory clear
nvidia-smi  # should show near-zero usage

# 4. Start dashboard
./dashboard.sh &

# 5. Launch training
python train.py
```

---

## Phase 3 — SmolLM2-135M Fast Iteration Track (new)

**Why add this:** Foundation-Sec-8B takes 4–12 hours per training run. You can't rapidly iterate on data quality, prompt formatting, or domain weights at that cadence. SmolLM2-135M (135M params) trains in **minutes**, has modern Llama-style architecture (RoPE, SwiGLU, GQA), and what you learn about data quality and instruction formatting transfers directly to Foundation-Sec-8B.

**Add to `config.yaml`:**
```yaml
# --- Fast Iteration Track ---
fast_iteration:
  model: "HuggingFaceTB/SmolLM2-135M-Instruct"
  max_seq_length: 512
  dtype: "bfloat16"
  attn_implementation: "flash_attention_2"
  
  lora:
    r: 8
    alpha: 16
    target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  
  training:
    epochs: 3
    batch_size: 32
    learning_rate: 2e-4
    output_dir: "outputs/smollm2_fast"
    logging_steps: 10
    time_limit_hours: 0.5   # Hard stop at 30 min
```

**Add `scripts/train_fast.py`:**
```python
"""
Fast iteration trainer using SmolLM2-135M.
Use this to validate data quality and prompt formats
before committing to a full Foundation-Sec-8B run.
"""
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import yaml

cfg = yaml.safe_load(open("config.yaml"))["fast_iteration"]

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=cfg["model"],
    max_seq_length=cfg["max_seq_length"],
    dtype=None,
    load_in_4bit=False,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=cfg["lora"]["r"],
    lora_alpha=cfg["lora"]["alpha"],
    target_modules=cfg["lora"]["target_modules"],
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing=True,
)

dataset = load_dataset(
    "json",
    data_files={"train": "data/export/chatml_train.jsonl"},
    split="train"
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=TrainingArguments(
        output_dir=cfg["training"]["output_dir"],
        per_device_train_batch_size=cfg["training"]["batch_size"],
        num_train_epochs=cfg["training"]["epochs"],
        learning_rate=cfg["training"]["learning_rate"],
        bf16=True,
        logging_steps=cfg["training"]["logging_steps"],
        report_to="tensorboard",
    ),
)
trainer.train()
model.save_pretrained("outputs/smollm2_fast/adapter")
print("Fast iteration run complete.")
```

**Fast iteration workflow:**
1. Change domain weights or add new data → `python scripts/train_fast.py` (30 min)
2. `python eval.py --model outputs/smollm2_fast/adapter` — check TTP accuracy
3. If accuracy improves → promote the data change to a full Foundation-Sec-8B run
4. Repeat until eval score plateaus, then run the full 4–12 hour Foundation-Sec-8B train

---

## Phase 4 — Qwen3.5-9B Game Adapter Track (existing, fix attention)

**Status:** Already designed in `config.yaml` under `game_adapters`. Pac-Man VLM using Qwen3.5-9B with visual inputs.

**Current config issue:**
```yaml
game_adapters:
  attention: "sdpa"   # suboptimal — fix to flash_attention_2 after sm_120 build
```

**After the Flash Attention build:**
```yaml
game_adapters:
  attention: "flash_attention_2"
```

**Pac-Man system prompt is already well-written:**
```
You are TARS, playing Pac-Man. Analyze the game frame and choose an action.
ACTIONS: NONE, UP, DOWN, LEFT, RIGHT
Respond with brief reasoning followed by: Action: <ACTION>
```

This is directly relevant to TARS robotics — the visual decision-making loop (observe frame → reason → act) is the same architecture you want for robot navigation.

---

## Phase 5 — Defender RL Track (existing, already production-quality)

**`defender.py` is already excellent.** It has:
- PPO with CNN policy via Stable Baselines3
- 8 parallel Atari environments
- Live dashboard streaming via httpx POST
- Graceful SIGINT handling (finishes rollout before saving)
- Best model checkpoint saving
- JSONL training log
- GPU temperature monitoring via pynvml

**No changes needed.** Just run it:
```bash
python defender.py train                    # 1M steps (default)
python defender.py train --steps 2000000   # 2M steps
python defender.py play                     # watch best model play live
```

**Connection to TARS:** The Defender game (protect humans from alien attack waves) is a useful RL environment because it requires spatial awareness and priority triage — skills that map to MDR alert triage. The agent must decide which threats to intercept and which to ignore, exactly like a SOC analyst deciding which alerts to escalate.

---

## Phase 6 — TurboQuant KV Cache Compression (Q3 2026)

**Status:** Not yet available (presenting at ICLR 2026). Add when released.

**What it does:** Compresses the KV cache to 3–4 bits with zero accuracy loss and zero retraining. For your Qwen3.5-9B with 262K context window, this means:

- 6x reduction in KV cache memory
- 8x speedup in attention logit computation vs FP32
- Long-context sessions (TARS memory, Splunk log analysis) become dramatically faster
- More concurrent sessions fit in 128GB

**Implementation (when released):**
```python
# Drop-in replacement for standard attention in vLLM config
from turboquant import TurboQuantAttention  # hypothetical import

# In vLLM serving config:
--kv-cache-dtype turboquant-4bit   # or however they expose it
```

**Watch:** `https://arxiv.org/abs/2504.19874` — when a vLLM PR merges support, add it to the inference stack.

---

## Data Pipeline Summary

```
mdr-log-simulator (192.168.1.67:6971)
    └── scripts/export_simulator_data.py
            └── data/export/chatml_train.jsonl
                    └── mixed into training via domain_weights

pentestds pipeline
    └── data/datasets/dist/chatml_train.jsonl
            └── CVE / MITRE / ExploitDB (245K examples)
                    └── filtered by quality.thresholds (Claude Sonnet 4-6 review)
                            └── final training set → Unsloth SFTTrainer
```

**Quality gate (already in config.yaml):**
```yaml
quality:
  review:
    model: "claude-sonnet-4-6"
    batch_size: 5
  thresholds:
    siem_queries: 4.0     # SPL must be syntactically valid
    sigma_rules: 4.0      # YAML must be valid
    secure_code_review: 4.0
    default: 3.5
```

---

## Inference Stack (post-training)

```bash
# Environment setup
export TORCH_CUDA_ARCH_LIST="12.0"
export VLLM_MOE_KERNEL_BACKEND=triton
export VLLM_DISABLED_KERNELS=cutlass_moe_mm,cutlass_scaled_mm
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH}"
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRITON_PTXAS_PATH=/usr/local/cuda-13.0/bin/ptxas

# Serve Foundation-Sec-8B fine-tuned model
python -m vllm.entrypoints.openai.api_server \
    --model ./outputs/threat-detector-v1 \
    --host 0.0.0.0 --port 8080 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --served-model-name foundation-sec-mdr

# Serve Qwen3.5-9B (TARS / game adapter)
python -m vllm.entrypoints.openai.api_server \
    --model ./adapters/games/pacman \
    --host 0.0.0.0 --port 8081 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.85 \
    --served-model-name tars-qwen
```

---

## Priority Order

| Priority | Task | Est. Time | Impact |
|---|---|---|---|
| 🔴 **P0** | Build Flash Attention for sm_120 | 75 min (one-time) | Unlocks full training throughput |
| 🔴 **P0** | Set CUDA 13/12 LD_LIBRARY_PATH | 5 min | Prevents runtime crashes |
| 🟠 **P1** | Run nanoGPT Shakespeare | 1 afternoon | Hyperparameter intuition |
| 🟠 **P1** | Run autoresearch overnight | 1 night | Architecture search insights |
| 🟡 **P2** | Add `scripts/export_simulator_data.py` | 2–3 hours | Live MDR data in training |
| 🟡 **P2** | Add `scripts/train_fast.py` (SmolLM2) | 2 hours | Fast iteration loop |
| 🟢 **P3** | First full Foundation-Sec-8B run | 4–12 hours | Core model output |
| 🟢 **P3** | Fix game adapter attention to FA2 | 5 min | Qwen3.5-9B training speed |
| 🔵 **P4** | TurboQuant KV compression | When released | Inference efficiency |

---

## Key Files Reference

| File | Purpose | Status |
|---|---|---|
| `config.yaml` | Central config — all hyperparams | ✅ Exists, needs FA fix |
| `train.py` | Foundation-Sec-8B SFT | ✅ Exists |
| `train_native.py` | Native DGX path | ✅ Exists |
| `train_game_adapter.py` | Qwen3.5-9B Pac-Man VLM | ✅ Exists |
| `defender.py` | PPO RL on Atari Defender | ✅ Exists, production-ready |
| `eval.py` | Model evaluation | ✅ Exists |
| `calibrate.py` | Temperature calibration | ✅ Exists |
| `dashboard.sh` | Launch monitoring UI on :28000 | ✅ Exists |
| `bench_throughput.py` | Throughput benchmarking | ✅ Exists |
| `scripts/export_simulator_data.py` | MDR simulator → ChatML | ❌ **Needs building** |
| `scripts/train_fast.py` | SmolLM2 fast iteration | ❌ **Needs building** |
| `docs/CLAUDE.md` | AI agent guidance | ❓ Check `docs/` folder |

---

## Resources

- **Base model:** huggingface.co/fdtn-ai/Foundation-Sec-8B-Instruct
- **Dataset:** huggingface.co/datasets/jason-oneal/mitre-stix-cve-exploitdb-dataset-alpaca-chatml-harmony
- **Dataset builder:** github.com/jason-allen-oneal/pentest-dataset-builder
- **Unsloth DGX Spark guide:** docs.unsloth.ai/basics/fine-tuning-llms-with-nvidia-dgx-spark-and-unsloth
- **Flash Attention sm_120 fix:** medium.com/@rakshith.d26/flash-attention-on-sm-121-solving-pytorch-compatibility-on-blackwell-gb10
- **autoresearch:** github.com/karpathy/autoresearch
- **TurboQuant paper:** arxiv.org/abs/2504.19874 (ICLR 2026)
- **NVD API key:** nvd.nist.gov/developers/request-an-api-key

---

*Plan generated from: file listing, dgx-spark-fine-tuning-plan.md, config.yaml, defender.py*
*Repo: github.com/fishkiler/fine-tune-mdr*
