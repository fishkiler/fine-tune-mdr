#!/bin/bash
# ============================================================================
# Fine-Tune MDR — Vast.ai H100 SXM Training Wrapper
# ============================================================================
# Self-contained setup + training script for a Vast.ai SSH instance.
#
# Prerequisites:
#   - Vast.ai instance with H100 SXM (80GB), PyTorch image
#   - Project files uploaded to /workspace/fine-tune-mdr/
#   - HF_TOKEN env var set (write-access token for private repo upload)
#
# Usage:
#   bash vast_train.sh              # Full pipeline: install, prepare, clean, train, eval, upload
#   bash vast_train.sh --skip-prep  # Skip data prep (if data/cleaned already exists)
#   bash vast_train.sh --eval-only  # Only run evaluation + upload
#   bash vast_train.sh --no-upload  # Skip HuggingFace upload
# ============================================================================

set -euo pipefail

PROJECT_DIR="/workspace/fine-tune-mdr"
CLEANED_DATA_DIR="data/cleaned"
ADAPTER_OUTPUT="outputs"
EVAL_EXAMPLES=500
HF_REPO="multiviper/foundation-sec-8b-mdr-lora-v2"

# --- Parse args ---
SKIP_PREP=false
EVAL_ONLY=false
NO_UPLOAD=false
for arg in "$@"; do
    case $arg in
        --skip-prep) SKIP_PREP=true ;;
        --eval-only) EVAL_ONLY=true; SKIP_PREP=true ;;
        --no-upload) NO_UPLOAD=true ;;
    esac
done

cd "$PROJECT_DIR"

echo "============================================"
echo "Fine-Tune MDR — Vast.ai H100 Training"
echo "============================================"
echo "Working directory: $(pwd)"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo ""

# ============================================================================
# Step 1: Install dependencies
# ============================================================================
echo "[1/7] Installing dependencies..."
pip install -q \
    "transformers>=4.57" \
    "peft>=0.18" \
    "trl>=0.23" \
    "datasets" \
    "accelerate" \
    "pyyaml" \
    "rouge-score" \
    "python-dotenv" \
    "huggingface_hub" \
    "flash-attn" --no-build-isolation 2>/dev/null

echo "Dependencies installed."

# --- Authenticate with HuggingFace ---
if [ -n "${HF_TOKEN:-}" ]; then
    echo "Logging in to HuggingFace..."
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
else
    echo "WARNING: HF_TOKEN not set. Set it for private repo upload:"
    echo "  export HF_TOKEN=hf_your_token_here"
fi

# ============================================================================
# Step 2: Prepare data (download from HuggingFace + format)
# ============================================================================
if [ "$SKIP_PREP" = false ]; then
    if [ ! -d "data/train" ]; then
        echo ""
        echo "[2/7] Preparing dataset..."
        python scripts/prepare_data.py --config config.yaml
    else
        echo ""
        echo "[2/7] Prepared data already exists at data/train — skipping."
    fi

    # ============================================================================
    # Step 3: Clean data
    # ============================================================================
    if [ ! -d "$CLEANED_DATA_DIR/train" ]; then
        echo ""
        echo "[3/7] Cleaning dataset..."
        python scripts/clean_data.py --config config.yaml --output "$CLEANED_DATA_DIR"
    else
        echo ""
        echo "[3/7] Cleaned data already exists at $CLEANED_DATA_DIR — skipping."
    fi
else
    echo ""
    echo "[2/7] Skipping data preparation (--skip-prep)."
    echo "[3/7] Skipping data cleaning (--skip-prep)."
fi

# Verify cleaned data exists
if [ ! -d "$CLEANED_DATA_DIR/train" ]; then
    echo "ERROR: Cleaned training data not found at $CLEANED_DATA_DIR/train"
    echo "Run without --skip-prep to prepare data first."
    exit 1
fi

# ============================================================================
# Step 4: Train
# ============================================================================
if [ "$EVAL_ONLY" = false ]; then
    echo ""
    echo "[4/7] Starting training..."
    echo "Config: $(python -c "import yaml; c=yaml.safe_load(open('config.yaml')); t=c['training']; print(f'epochs={t[\"num_epochs\"]}, batch={t[\"per_device_train_batch_size\"]}, lr={t[\"learning_rate\"]}, warmup={t[\"warmup_steps\"]}')")"
    echo ""

    python train_native.py \
        --config config.yaml \
        --data-dir "$CLEANED_DATA_DIR" \
        --fresh

    echo ""
    echo "Training complete."
else
    echo ""
    echo "[4/7] Skipping training (--eval-only)."
fi

# ============================================================================
# Step 5: Run evaluation
# ============================================================================
echo ""
echo "[5/7] Running evaluation on $EVAL_EXAMPLES test examples..."
python eval.py --config config.yaml --model-dir "$ADAPTER_OUTPUT"

# ============================================================================
# Step 6: Upload adapter to HuggingFace (private)
# ============================================================================
if [ "$NO_UPLOAD" = false ] && [ -n "${HF_TOKEN:-}" ]; then
    echo ""
    echo "[6/7] Uploading adapter to HuggingFace (private repo: $HF_REPO)..."
    python -c "
from huggingface_hub import HfApi
import json, os

api = HfApi()

# Create private repo (no-op if exists)
api.create_repo('${HF_REPO}', private=True, exist_ok=True)

# Upload adapter files
api.upload_folder(
    folder_path='${ADAPTER_OUTPUT}',
    repo_id='${HF_REPO}',
    ignore_patterns=[
        'checkpoint-*',
        'logs/*',
        'exported/*',
        '*.bin',
        'training_args.bin',
    ],
    commit_message='Upload LoRA adapter from Vast.ai H100 training run',
)

# Upload eval results
if os.path.isdir('eval_results'):
    api.upload_folder(
        folder_path='eval_results',
        repo_id='${HF_REPO}',
        path_in_repo='eval_results',
        commit_message='Upload evaluation results',
    )

print(f'Uploaded to https://huggingface.co/${HF_REPO} (private)')
"
    echo "Upload complete."
else
    if [ "$NO_UPLOAD" = true ]; then
        echo ""
        echo "[6/7] Skipping upload (--no-upload)."
    else
        echo ""
        echo "[6/7] Skipping upload — HF_TOKEN not set."
    fi
fi

# ============================================================================
# Step 7: Summary
# ============================================================================
echo ""
echo "[7/7] Done!"
echo "============================================"
echo "Outputs:"
echo "  Adapter:     $ADAPTER_OUTPUT/"
echo "  Results:     eval_results/"
if [ "$NO_UPLOAD" = false ] && [ -n "${HF_TOKEN:-}" ]; then
echo "  HuggingFace: https://huggingface.co/$HF_REPO (private)"
fi
echo "============================================"
echo ""
echo "To download adapter locally:"
echo "  pip install huggingface_hub"
echo "  huggingface-cli login"
echo "  huggingface-cli download $HF_REPO --local-dir ./adapter-v2"
echo ""
echo "Or via SCP:"
echo "  scp -P <port> -r root@<host>:$PROJECT_DIR/$ADAPTER_OUTPUT/ ./outputs/"
echo "  scp -P <port> root@<host>:$PROJECT_DIR/eval_results/*.json ./eval_results/"
