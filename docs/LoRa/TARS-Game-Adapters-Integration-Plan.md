# TARS Game Adapters — Integration Plan for Fine-Tune MDR

**Date:** March 11, 2026  
**Target Application:** Fine-Tune MDR (DGX Spark, 192.168.1.205)  
**First Adapter:** Pac-Man (original arcade via MAME)  
**Base VLM:** Qwen3.5-9B  
**Status:** Planning — ready for implementation

---

## 1. What We Are Doing

We are extending the Fine-Tune MDR application to support a second class of fine-tuning: **vision-language game adapters**. The existing application fine-tunes Foundation-Sec-8B (a text-only cybersecurity LLM) using LoRA on curated threat intelligence data. We are adding the ability to fine-tune Qwen3.5-9B (a multimodal vision-language model) with LoRA adapters that teach TARS to play specific video games by looking at screen frames and choosing actions.

The first adapter is for **Pac-Man** — the original 1980 Namco arcade game running in MAME. Jay is recording gameplay through MAME and capturing every frame alongside the corresponding joystick inputs. The current dataset exceeds 80,000 labeled frame-action pairs and is still growing.

Each game gets its own independent LoRA adapter — a small ~200–400MB file that bolts onto the frozen Qwen3.5-9B base model. At inference time, TARS loads the base model once and hot-swaps the appropriate game adapter on demand. This means adding a new game costs only the training time and adapter storage, not another full model copy.

**Why this lives inside Fine-Tune MDR rather than as a separate project:** The MDR application already solves all the hard infrastructure problems — cgroup-guarded training on the DGX Spark, a real-time training dashboard with SSE metrics streaming, config-driven LoRA hyperparameters, SQLite-backed dataset management with validation and export pipelines, and evaluation tooling. Rather than rebuilding all of that from scratch, we add a new training profile to the existing system. The game adapter pipeline follows the same lifecycle stages the MDR pipeline does: ingest → validate → export → train → evaluate → serve. The implementation details differ (images instead of text, VLM instead of text LLM, action accuracy instead of technique accuracy), but the orchestration patterns are identical.

**Future games** will follow the exact same pipeline. Once the Pac-Man adapter is working, adding a new game means: record MAME gameplay → run the ingestion script pointing at the new game's frame directory → train with `--game galaga` (or whatever the game is) → deploy the adapter. The infrastructure, training script, dashboard integration, database schema, and adapter registry all support multiple games from day one.

---

## 2. How It Connects to Qwen3.5-9B

The MDR application currently trains one model: Foundation-Sec-8B-Instruct (Llama 3.1 architecture, text-only). Game adapters introduce a second base model: Qwen3.5-9B (multimodal, accepts images + text).

These are completely independent training targets. They share the DGX Spark hardware, the training dashboard, the cgroup wrapper, and the config-driven approach, but they use different model loading code, different data formats, and different evaluation metrics. The `config.yaml` gains a new top-level section (`game_adapters`) alongside the existing MDR training config, and a new training script (`train_game_adapter.py`) sits beside `train_native.py`.

At inference time on the TARS robot, the two models serve different purposes entirely. Foundation-Sec-8B handles cybersecurity analysis through the MDR inference server. Qwen3.5-9B is TARS's general vision-language model — it already runs on the DGX Spark for vision analysis, conversation, and reasoning. The game LoRA adapters temporarily specialize that same Qwen3.5-9B for gameplay. When TARS is not playing a game, the adapter is unloaded and Qwen3.5-9B returns to its normal role.

```
┌─────────────────────────────────────────────────────────┐
│                    DGX Spark (192.168.1.205)             │
│                                                         │
│  ┌─────────────────────┐  ┌──────────────────────────┐  │
│  │  Foundation-Sec-8B   │  │  Qwen3.5-9B              │  │
│  │  (MDR fine-tune)     │  │  (TARS general VLM)      │  │
│  │                      │  │                          │  │
│  │  LoRA: MDR adapter   │  │  LoRA: none (default)    │  │
│  │  Serves: /analyze    │  │    or: pacman_v1         │  │
│  │  Port: TBD           │  │    or: galaga_v1         │  │
│  └─────────────────────┘  │    or: (future game)      │  │
│                            │  Serves: /game/action     │  │
│                            │  Port: 41993              │  │
│                            └──────────────────────────┘  │
│                                                         │
│  ┌─────────────────────────────────────────────────────┐ │
│  │  Shared Infrastructure                               │ │
│  │  • Training Dashboard (port 28000)                   │ │
│  │  • SQLite databases                                  │ │
│  │  • train.sh cgroup wrapper (MemoryMax=110G)          │ │
│  │  • config.yaml                                       │ │
│  └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

---

## 3. Data Pipeline — MAME Frame Ingestion

### 3.1 What We Have

Jay is recording Pac-Man gameplay through MAME, capturing every frame as an image file with the corresponding joystick input labeled alongside it. The current dataset is 80,000+ frames and growing.

### 3.2 New Ingestion Script

A new script `scripts/sources/ingest_mame_frames.py` handles importing MAME recordings into the database. It follows the same patterns as the existing source scripts (`ingest_cveorg.py`, `fetch_cisa_kev.py`, etc.) — deduplication by content hash, incremental ingestion, and integration with the shared `db_utils.py` layer.

The script:

1. Scans a MAME recording directory for frame images and their action labels
2. Parses the action labels from whatever format MAME exports (likely a log file mapping frame numbers to joystick inputs, or filenames encoding the action)
3. Hashes each frame for deduplication (perceptual hash, not cryptographic — near-duplicate frames from consecutive identical states get deduplicated)
4. Inserts records into a new `game_frames` table with: frame path, action label, game name, episode ID, frame index within episode, cumulative score if available
5. Reports statistics: total frames ingested, duplicates skipped, action distribution

```bash
# Usage pattern (follows existing conventions)
python scripts/sources/ingest_mame_frames.py \
    --game pacman \
    --frames-dir /path/to/mame/recordings/pacman/ \
    --action-format mame_inp  # or csv, or filename-encoded
```

### 3.3 Database Schema Addition

New tables added to the existing SQLite database (or a separate `game_adapters.db` if you prefer isolation). This would be a schema v6 migration:

```sql
-- Game adapter metadata
CREATE TABLE game_adapters (
    id INTEGER PRIMARY KEY,
    game_name TEXT NOT NULL UNIQUE,        -- 'pacman', 'galaga', etc.
    base_model TEXT NOT NULL,               -- 'Qwen/Qwen3.5-9B'
    adapter_path TEXT,                      -- path to trained adapter
    adapter_version INTEGER DEFAULT 0,
    frame_resolution TEXT,                  -- '288x224' (MAME Pac-Man native)
    action_space TEXT NOT NULL,             -- JSON list of valid actions
    status TEXT DEFAULT 'collecting',       -- collecting | training | trained | deployed
    total_frames INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    trained_at TIMESTAMP,
    eval_score_avg REAL,                   -- average score across eval games
    eval_score_best REAL,
    notes TEXT
);

-- Individual frames with action labels
CREATE TABLE game_frames (
    id INTEGER PRIMARY KEY,
    game_name TEXT NOT NULL,
    frame_path TEXT NOT NULL,               -- relative path to PNG
    frame_hash TEXT NOT NULL,               -- perceptual hash for dedup
    action_label TEXT NOT NULL,             -- 'UP', 'LEFT', 'NOOP', etc.
    action_id INTEGER NOT NULL,             -- numeric action index
    episode_id INTEGER,                     -- which gameplay session
    frame_index INTEGER,                    -- position within episode
    cumulative_score REAL,                  -- score at this frame
    quality_score REAL,                     -- optional quality rating
    validated INTEGER DEFAULT 0,            -- passed validation checks
    excluded INTEGER DEFAULT 0,             -- manually excluded
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (game_name) REFERENCES game_adapters(game_name)
);

-- Training run history (parallels export_history)
CREATE TABLE game_training_runs (
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
    config_snapshot TEXT,                   -- JSON dump of full training config
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    FOREIGN KEY (game_name) REFERENCES game_adapters(game_name)
);

CREATE INDEX idx_game_frames_game ON game_frames(game_name);
CREATE INDEX idx_game_frames_hash ON game_frames(frame_hash);
CREATE INDEX idx_game_frames_episode ON game_frames(game_name, episode_id);
```

### 3.4 Pac-Man Action Space

The original Pac-Man arcade has a 4-direction joystick (no diagonals, no NOOP-as-neutral — Pac-Man continues in his current direction if no input is given). The action space is:

```python
PACMAN_ACTIONS = {
    "game": "pacman",
    "actions": [
        {"id": 0, "name": "NONE",  "description": "No joystick input — continue current direction"},
        {"id": 1, "name": "UP",    "description": "Move up"},
        {"id": 2, "name": "DOWN",  "description": "Move down"},
        {"id": 3, "name": "LEFT",  "description": "Move left"},
        {"id": 4, "name": "RIGHT", "description": "Move right"},
    ]
}
```

This may need adjustment based on exactly how MAME encodes joystick inputs in your recording format.

### 3.5 Validation

A new validation function in `scripts/validate_data.py` (or a separate `scripts/validate_game_data.py`) checks game frames for:

- **Action balance** — flag if any single action exceeds 60% of the dataset (NONE/NOOP often dominates and needs downsampling)
- **Corrupt frames** — verify each PNG opens and has the expected resolution
- **Duplicate detection** — perceptual hashing catches near-identical frames that waste training budget
- **Episode continuity** — frame indices within each episode should be sequential without gaps
- **Score progression** — if scores are recorded, they should be non-decreasing within an episode (except on death)

---

## 4. Dataset Formatting — VLM Conversation Pairs

### 4.1 Export Script

A new export script `scripts/export_game_training_data.py` converts the raw `game_frames` table into the Qwen3.5 VLM conversation format. This parallels `scripts/export_training_data.py` for the MDR pipeline.

Each frame becomes a multi-turn conversation:

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are TARS, playing Pac-Man. Analyze the game frame and choose an action.\n\nACTIONS: NONE (continue current direction), UP, DOWN, LEFT, RIGHT\n\nRespond with brief reasoning followed by: Action: <ACTION>"
    },
    {
      "role": "user",
      "content": [
        {"type": "image", "image": "frames/pacman/frame_000042.png"},
        {"type": "text", "text": "What action should Pac-Man take?"}
      ]
    },
    {
      "role": "assistant",
      "content": "Pac-Man is heading right in a corridor with pellets ahead. The path is clear and a ghost is visible in the upper corridor moving away. Continuing right to collect pellets.\n\nAction: RIGHT"
    }
  ]
}
```

### 4.2 Chain-of-Thought Reasoning Generation

The raw MAME data only has frame→action pairs — there's no reasoning text. The export script needs to generate reasoning for the assistant responses. Three strategies, in order of quality:

**Strategy A — Template-based (fast, free, good enough to start):**
Pre-written reasoning templates keyed by action, randomly selected. Gets training running immediately. This is what you should use for v1.

**Strategy B — VLM-annotated (better quality, costs API credits):**
Send each frame to a frontier VLM (Qwen3-VL-30B via OpenRouter, or Claude) with the known-correct action and ask it to generate a 2-3 sentence explanation of why that action is correct. Produces richer, frame-specific reasoning. Can be done in batch overnight.

**Strategy C — No reasoning (simplest):**
Skip reasoning entirely, train the assistant to output only `Action: RIGHT`. Faster training, smaller sequence lengths, but the model learns less about *why* it's choosing actions and can't narrate gameplay.

Recommendation: start with Strategy A for the first training run, then upgrade to Strategy B for v2 once you've confirmed the pipeline works end to end.

### 4.3 Action Balancing

Pac-Man gameplay is heavily skewed toward movement actions and light on direction changes (Pac-Man often travels long corridors). The export script applies the same domain-weighting concept the MDR pipeline uses:

- If NONE/NOOP exceeds 40% of frames, downsample to 25%
- Oversample frames where direction changes occur (these are the decision points)
- Oversample frames near ghosts (high-stakes decisions)
- Oversample frames near power pellets (strategic moments)

The ghost-proximity and power-pellet logic requires frame analysis at export time (or pre-computed labels during ingestion). A simple heuristic: frames where the action changes from the previous frame's action are decision points and get upsampled 3x.

---

## 5. Training Script — `train_game_adapter.py`

### 5.1 Design

A new training script that parallels `train_native.py` but handles multimodal VLM training. Key differences from the MDR training script:

| Aspect | `train_native.py` (MDR) | `train_game_adapter.py` (Games) |
|--------|--------------------------|----------------------------------|
| Base model | Foundation-Sec-8B-Instruct | Qwen3.5-9B |
| Model class | AutoModelForCausalLM | AutoModelForVision2Seq |
| Input format | Text (instruction/response) | Image + text conversation |
| Processor | AutoTokenizer | AutoProcessor (handles images) |
| Attention | SDPA | SDPA (same — no Flash Attn on Spark) |
| LoRA targets | Attention + MLP projections | Attention + MLP projections (same) |
| Metrics POST | `http://localhost:28000/log` | `http://localhost:28000/log` (same) |
| Cgroup wrapper | `train.sh` | `train_game.sh` (same pattern) |
| Output | `adapters/mdr/` | `adapters/games/pacman_v1/` |

### 5.2 Config Section

New section in `config.yaml`:

```yaml
# ─── Existing MDR config (unchanged) ───
model:
  name: "fdtn-ai/Foundation-Sec-8B-Instruct"
  # ... existing MDR settings ...

lora:
  r: 16
  alpha: 32
  # ... existing MDR LoRA settings ...

# ─── New: Game Adapter config ───
game_adapters:
  base_model: "Qwen/Qwen3.5-9B"
  attention: "sdpa"                      # No flash attention on DGX Spark
  precision: "bf16"                      # BF16 faster than 4-bit on Blackwell
  torch_compile: false                   # Causes recompilation spikes on Spark

  lora:
    r: 32
    alpha: 64
    dropout: 0.05
    target_modules:
      - "q_proj"
      - "k_proj"
      - "v_proj"
      - "o_proj"
      - "gate_proj"
      - "up_proj"
      - "down_proj"

  training:
    epochs: 5
    batch_size: 4                        # Per-device
    gradient_accumulation_steps: 4       # Effective batch = 16
    learning_rate: 2.0e-4
    lr_scheduler: "cosine"
    warmup_steps: 100
    weight_decay: 0.01
    max_seq_length: 2048
    gradient_checkpointing: true
    dataloader_num_workers: 4

  image:
    min_pixels: 200704                   # 256 * 28 * 28
    max_pixels: 401408                   # 512 * 28 * 28

  eval:
    num_games: 50                        # Games to play for evaluation
    frame_skip: 4                        # Repeat each action for N game frames

  games:
    pacman:
      display_name: "Pac-Man"
      mame_rom: "pacman"
      frame_resolution: "288x224"
      actions: ["NONE", "UP", "DOWN", "LEFT", "RIGHT"]
      system_prompt: >
        You are TARS, playing Pac-Man. Analyze the game frame and choose an action.
        ACTIONS: NONE (continue current direction), UP, DOWN, LEFT, RIGHT
        Respond with brief reasoning followed by: Action: <ACTION>
      dataset_dir: "data/games/pacman"
      adapter_dir: "adapters/games/pacman"
      min_score_threshold: 1000
      action_balance:
        max_noop_ratio: 0.25
        decision_point_oversample: 3

    # ─── Future games (same structure) ───
    # galaga:
    #   display_name: "Galaga"
    #   mame_rom: "galaga"
    #   frame_resolution: "288x224"
    #   actions: ["NONE", "LEFT", "RIGHT", "FIRE", "FIRE_LEFT", "FIRE_RIGHT"]
    #   system_prompt: ...
    #   dataset_dir: "data/games/galaga"
    #   adapter_dir: "adapters/games/galaga"

    # donkey_kong:
    #   display_name: "Donkey Kong"
    #   ...
```

### 5.3 Cgroup Wrapper

A new `train_game.sh` following the same pattern as `train.sh`:

```bash
#!/bin/bash
# train_game.sh — Cgroup-guarded game adapter training
# Prevents NVIDIA driver deadlocks on OOM

GAME="${1:?Usage: train_game.sh <game_name>}"

# Same cgroup pattern as train.sh
CGROUP="/sys/fs/cgroup/tars-game-train"
sudo mkdir -p "$CGROUP"
echo "110G" | sudo tee "$CGROUP/memory.max"

echo $$ | sudo tee "$CGROUP/cgroup.procs"

python train_game_adapter.py --game "$GAME"
```

```bash
# Usage
bash train_game.sh pacman
bash train_game.sh galaga    # future
```

### 5.4 Dashboard Integration

The training script POSTs metrics to the existing dashboard at `http://localhost:28000/log`, using the same SSE streaming the MDR training uses. The metrics payload includes a `training_type` field so the dashboard can distinguish between MDR and game adapter runs:

```python
# Inside train_game_adapter.py training loop
import httpx

def log_metrics(step, loss, lr, game_name):
    httpx.post("http://localhost:28000/log", json={
        "training_type": "game_adapter",
        "game": game_name,
        "step": step,
        "loss": loss,
        "learning_rate": lr,
    })
```

The dashboard frontend would need a small addition to display game adapter training runs alongside MDR runs — a tab or dropdown to switch between them, or a combined view with color-coded series.

### 5.5 Memory Budget (Qwen3.5-9B, bf16, DGX Spark)

| Component | Size |
|-----------|------|
| Qwen3.5-9B weights (bf16) | ~18 GB |
| LoRA adapters (r=32, bf16) | ~0.15 GB |
| Optimizer states (AdamW) | ~0.6 GB |
| Gradients | ~0.15 GB |
| Activations (batch=4, checkpointed) | ~8 GB |
| Image processing overhead | ~2 GB |
| CUDA overhead | ~2 GB |
| **Total peak** | **~31 GB** |
| **Cgroup limit** | **110 GB** |
| **Headroom** | **~79 GB** |

Well within the cgroup limit. You could increase batch size to 8 or even 16 if training is too slow, though image processing makes batches heavier than text-only training.

### 5.6 Estimated Training Time

Based on your DGX Spark's observed 890 tok/s throughput on Foundation-Sec-8B (text-only) and accounting for the image processing overhead of the VLM (roughly 2-3x slower per sample than text-only due to vision encoder forward passes):

| Dataset Size | Epochs | Estimated Time |
|-------------|--------|----------------|
| 40,000 frames (balanced from 80K) | 5 | ~3–5 hours |
| 60,000 frames | 5 | ~5–8 hours |
| 80,000 frames (full) | 5 | ~7–10 hours |
| 80,000 frames | 10 | ~14–20 hours |

Start with 5 epochs on the balanced dataset. If loss is still decreasing at epoch 5, extend to 8-10.

---

## 6. Evaluation — `eval_game_adapter.py`

### 6.1 Metrics

Game adapter evaluation is fundamentally different from MDR evaluation. Instead of exact match and F1 on text outputs, we measure actual gameplay performance:

| Metric | Description | How Measured |
|--------|-------------|-------------|
| **Average score** | Mean score across N games | Run inference loop, record final scores |
| **Best score** | Highest single-game score | Max of N games |
| **Survival time** | Average frames alive per game | Track death events |
| **Action accuracy** | Agreement with expert data on held-out test frames | Standard classification accuracy on 10% test split |
| **Action diversity** | Entropy of action distribution during play | Ensures model isn't stuck in a loop |
| **Decision quality** | Accuracy specifically on "decision point" frames | Measures performance on the hard cases |

### 6.2 Evaluation Script

`eval_game_adapter.py` runs alongside `eval.py` (MDR). It loads the adapter, plays N games, and records metrics into the `game_training_runs` table. For Pac-Man, this requires either a MAME scripting interface or a gym-compatible wrapper for the game.

Since MAME doesn't have a native Python gym interface like ALE does, evaluation can work in two modes:

**Mode A — Offline accuracy (no emulator needed):**
Hold out 10% of the labeled frames as a test set. Run the model on each test frame and compare the predicted action to the recorded action. This gives you action accuracy without needing to run a live game. This is the v1 approach.

**Mode B — Live gameplay (requires MAME scripting):**
Use MAME's Lua scripting interface or a bridge like `mame-ai` to drive the emulator in real-time with model-predicted actions. This gives you actual scores and survival time. This is the v2 approach once the pipeline is proven.

---

## 7. Adapter Registry & Hot-Swapping

### 7.1 Adapter Directory Structure

All game adapters follow a consistent directory layout under `adapters/games/`:

```
adapters/
├── mdr/                              # Existing MDR adapter
│   └── foundation-sec-lora/
└── games/                            # New: game adapters
    ├── registry.json                 # Adapter registry (see below)
    ├── pacman/
    │   ├── v1/
    │   │   ├── adapter_model.safetensors   # ~200-400MB
    │   │   ├── adapter_config.json
    │   │   ├── tokenizer/
    │   │   └── training_metadata.json      # Config snapshot, metrics
    │   ├── v2/                             # After retraining
    │   │   └── ...
    │   └── active -> v1/                   # Symlink to deployed version
    ├── galaga/                             # Future
    │   ├── v1/
    │   └── active -> v1/
    └── donkey_kong/                        # Future
        └── ...
```

### 7.2 Registry File

`adapters/games/registry.json` tracks all available game adapters:

```json
{
  "base_model": "Qwen/Qwen3.5-9B",
  "adapters": {
    "pacman": {
      "display_name": "Pac-Man",
      "active_version": "v1",
      "versions": {
        "v1": {
          "path": "adapters/games/pacman/v1",
          "trained_at": "2026-03-12T04:30:00Z",
          "training_frames": 42000,
          "epochs": 5,
          "eval_avg_score": 2450,
          "eval_best_score": 8200,
          "action_accuracy": 0.73,
          "lora_r": 32,
          "adapter_size_mb": 287
        }
      }
    }
  }
}
```

### 7.3 Hot-Swap at Inference Time

The game adapter inference server (port 41993) loads the base Qwen3.5-9B model once and swaps LoRA adapters on demand:

```python
# Conceptual — adapter swap logic inside the inference server
from peft import PeftModel, set_peft_model_state_dict

class GameAdapterServer:
    def __init__(self):
        self.base_model = load_qwen35_9b()
        self.active_adapter = None
        self.active_game = None

    def load_game(self, game_name: str):
        """Hot-swap to a different game adapter."""
        if self.active_game == game_name:
            return  # Already loaded

        registry = load_registry()
        adapter_path = registry["adapters"][game_name]["active_version_path"]

        # Unload current adapter if any
        if self.active_adapter:
            self.model = self.base_model  # Reset to base

        # Load new adapter
        self.model = PeftModel.from_pretrained(self.base_model, adapter_path)
        self.model.eval()
        self.active_game = game_name

    def predict_action(self, frame, game_name: str):
        """Given a game frame, predict the best action."""
        self.load_game(game_name)  # Ensure correct adapter is loaded
        # ... run inference, return action
```

TARS triggers adapter swaps through voice commands routed by the 17-stage intent chain:

```
"Hey TARS, play Pac-Man"     → load_game("pacman")
"TARS, switch to Galaga"     → load_game("galaga")
"Stop playing"               → unload adapter, return to base VLM
```

---

## 8. Integration Touchpoints — What Changes in Existing Code

This section lists every file in the Fine-Tune MDR application that needs modification, and what the modification is. New files are also listed.

### 8.1 Modified Files

| File | Change |
|------|--------|
| `config.yaml` | Add `game_adapters` section (Section 5.2 above) |
| `dashboard/server.py` | Accept `training_type: "game_adapter"` in `/log` POST, store in separate metrics series |
| `dashboard/static/index.html` | Add tab/toggle to switch between MDR and game adapter training views |
| `scripts/db_utils.py` | Add schema v6 migration with `game_adapters`, `game_frames`, `game_training_runs` tables |
| `mdr-database/build_dataset_db.py` | Add `--migrate` support for v6 schema |
| `mdr-database/view_dataset.py` | Add game frames browsing tab (filter by game, view frames + actions) |

### 8.2 New Files

| File | Purpose |
|------|---------|
| `train_game_adapter.py` | LoRA fine-tuning script for VLM game adapters |
| `train_game.sh` | Cgroup-guarded training wrapper for game adapters |
| `eval_game_adapter.py` | Evaluation: action accuracy on held-out test set + live gameplay metrics |
| `game_inference.py` | FastAPI server for game adapter inference (port 41993) |
| `scripts/sources/ingest_mame_frames.py` | MAME frame + action label ingestion |
| `scripts/validate_game_data.py` | Game-specific validation (action balance, frame integrity, dedup) |
| `scripts/export_game_training_data.py` | Export frames to Qwen3.5 conversation format with action balancing |
| `adapters/games/registry.json` | Adapter registry tracking all trained game adapters |

### 8.3 New Directories

```
data/games/pacman/              # MAME frame data for Pac-Man
data/games/pacman/frames/       # PNG frame images
data/games/pacman/training/     # Exported VLM conversation JSONL
adapters/games/                 # Trained adapter storage
adapters/games/pacman/          # Pac-Man adapter versions
```

---

## 9. Pipeline Commands

These follow the same patterns as the existing MDR pipeline commands:

```bash
# ─── Data Ingestion ───
# Ingest MAME Pac-Man recordings
python scripts/sources/ingest_mame_frames.py \
    --game pacman \
    --frames-dir /path/to/mame/pacman/frames/

# Validate frame data
python scripts/validate_game_data.py --game pacman

# Export to VLM training format (with action balancing)
python scripts/export_game_training_data.py --game pacman

# ─── Training ───
# Train Pac-Man adapter (cgroup-guarded)
bash train_game.sh pacman

# ─── Evaluation ───
# Evaluate on held-out test frames
python eval_game_adapter.py --game pacman --mode offline

# ─── Inference ───
# Start game adapter server
python game_inference.py --port 41993

# ─── Future games (same commands, different --game flag) ───
# python scripts/sources/ingest_mame_frames.py --game galaga --frames-dir /path/to/galaga/
# bash train_game.sh galaga
# python eval_game_adapter.py --game galaga --mode offline
```

---

## 10. Implementation Phases

### Phase 1 — Foundation (get training running)

1. Add `game_adapters` section to `config.yaml`
2. Add schema v6 migration to `db_utils.py`
3. Write `scripts/sources/ingest_mame_frames.py` — get the 80,000+ Pac-Man frames into the database
4. Write `scripts/export_game_training_data.py` — format frames into Qwen3.5 conversation JSONL with template-based reasoning (Strategy A)
5. Write `train_game_adapter.py` — the core training script, POSTing metrics to dashboard
6. Write `train_game.sh` — cgroup wrapper
7. Run first training on Pac-Man dataset
8. Verify adapter saves correctly and loads back onto Qwen3.5-9B

### Phase 2 — Evaluation & Iteration

9. Write `eval_game_adapter.py` — offline action accuracy on held-out test set
10. Analyze action distribution and error patterns
11. Add action balancing to export script (downsample NONE, oversample decision points)
12. Retrain with balanced dataset → compare eval metrics
13. Add game adapter training runs to `game_training_runs` table

### Phase 3 — Inference & TARS Integration

14. Write `game_inference.py` — FastAPI server with adapter hot-swapping on port 41993
15. Create `adapters/games/registry.json`
16. Integrate with TARS intent router — voice command triggers game mode
17. Wire up to TARS HUD — display game on the 800×480 kiosk display, overlay TARS commentary

### Phase 4 — Dashboard & Polish

18. Update `dashboard/server.py` to handle `training_type: "game_adapter"` metrics
19. Add game adapter tab to dashboard frontend
20. Update `view_dataset.py` with game frame browsing
21. Write `scripts/validate_game_data.py` for frame-specific quality checks

### Phase 5 — Future Games

22. Record gameplay for next game (Galaga, Donkey Kong, Space Invaders, etc.)
23. Run the same pipeline: `ingest → validate → export → train → eval → deploy`
24. Each game gets its own entry in `config.yaml` under `game_adapters.games`
25. Each game gets its own LoRA adapter (~200-400MB) sharing the same Qwen3.5-9B base
26. The adapter registry and hot-swap server handle switching between games automatically

---

## 11. Adding a New Game (Future Reference)

Once the Pac-Man pipeline is working, adding any new MAME game follows this checklist:

```
[ ] 1. Record gameplay in MAME with frame + input capture
[ ] 2. Add game entry to config.yaml under game_adapters.games
        - Define action space, frame resolution, system prompt
[ ] 3. Run: python scripts/sources/ingest_mame_frames.py --game <name> --frames-dir <path>
[ ] 4. Run: python scripts/validate_game_data.py --game <name>
[ ] 5. Run: python scripts/export_game_training_data.py --game <name>
[ ] 6. Run: bash train_game.sh <name>
[ ] 7. Run: python eval_game_adapter.py --game <name> --mode offline
[ ] 8. Update adapters/games/registry.json with new adapter
[ ] 9. Restart game_inference.py — new game is automatically available
[ ] 10. TARS can now play the new game via voice command
```

Total effort per new game: recording time + ~30 minutes of commands + training time (3-10 hours depending on dataset size).

---

## 12. Port Map (Updated)

| Port | Service | Status |
|------|---------|--------|
| 28000 | Fine-Tune MDR Training Dashboard | Existing |
| 11969 | STIX Graph Visualization | Existing |
| 8080 | Dataset Viewer | Existing |
| 41988 | TARS Main HUD/Web | Existing |
| 41989 | TTS Gateway | Existing |
| 41990 | LLM API Proxy | Existing |
| 41991 | Reserved | Existing |
| 41992 | Brain Telemetry WebSocket | Existing |
| **41993** | **Game Adapter Inference Server** | **New** |
