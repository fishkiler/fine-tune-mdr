# DGX Spark Fine-Tuning Reference: Llama 3.1 70B with QLoRA

> Source: Sanjay Basu PhD — "Fine-Tuning Llama 3.1 70B on DGX SPARK" (Oct 2025)
> Purpose: Troubleshooting lockups and optimizing fine-tuning on DGX Spark (128GB unified memory)

---

## CRITICAL: System Lockup Diagnosis

If the DGX Spark is locking up during fine-tuning, the most likely causes are:

1. **Memory exhaustion in unified memory** — CPU and GPU share the 128GB pool. If peak usage exceeds available memory (after OS/CUDA overhead), the system will freeze or OOM-kill.
2. **Activation memory blowout** — Long sequences without gradient checkpointing consume ~5.2GB per sample per forward pass.
3. **CUDA memory fragmentation** — Repeated allocations without cleanup cause fragmentation on ARM64.

---

## Memory Budget (Target: Stay Under ~90GB to Leave Headroom)

| Component | Size |
|---|---|
| Model Weights (4-bit quantized) | 35.0 GB |
| LoRA Adapters (FP16) | 0.2 GB |
| Optimizer States (AdamW) | 0.8 GB |
| Gradients | 0.2 GB |
| Activations (per sample, with checkpointing) | ~4.0 GB |
| CUDA Overhead | ~2.0 GB |
| **Total Peak** | **~42.2 GB** |

If you're seeing lockups, actual peak is likely much higher than 42GB. Common causes:
- Gradient checkpointing is **not enabled** or not working correctly
- Sequence length is too long (8192+ tokens without Flash Attention)
- CUDA memory allocator is fragmenting (see settings below)
- Other processes consuming unified memory (check with `free -h` and `nvidia-smi`)

---

## Recommended Hyperparameters for Stability

```yaml
# Conservative settings to prevent lockups
micro_batch_size: 1
gradient_accumulation_steps: 4      # Reduce from 8 if unstable
sequence_len: 4096                   # Reduce from 8192 if OOM
gradient_checkpointing: true         # ESSENTIAL — saves 3-4x memory
flash_attention: true                # ESSENTIAL for long contexts
bf16: true                           # Better than fp16 on Blackwell
optimizer: paged_adamw_8bit          # Paged optimizer saves memory
learning_rate: 0.0002
warmup_steps: 100
lora_r: 16
lora_alpha: 32
weight_decay: 0.01
```

### If Still Locking Up — Progressive Reduction

Try these changes **in order**, testing after each:

1. `sequence_len: 2048` (down from 4096/8192)
2. `lora_r: 8` (down from 16)
3. `gradient_accumulation_steps: 2` (down from 4/8)
4. `flash_attention: false` (some ARM64 builds have buggy FA2)
5. Disable W&B logging if enabled (reduces memory overhead)

---

## Blackwell / DGX Spark CUDA Settings

These should be set **before** training starts:

```python
import torch
import os

# Enable TF32 for faster matmuls (Blackwell supports this)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# CRITICAL for preventing memory fragmentation lockups
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

# Enable Flash Attention 2 (if supported)
# model.config.use_flash_attention_2 = True
```

### Environment Variables to Set Before Launch

```bash
# Memory allocator tuning — prevents fragmentation-induced lockups
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# If lockups persist, try larger split size
# export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

---

## ARM64-Specific Issues

The DGX Spark uses ARM architecture. Known problems:

- **Random CUDA errors/crashes**: Update drivers with `sudo apt update && sudo apt upgrade nvidia-driver-535`
- **PyTorch build mismatch**: Ensure ARM64-native PyTorch is installed:
  ```bash
  pip uninstall torch torchvision torchaudio
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
  ```
- **Flash Attention compatibility**: If crashes occur with FA enabled, disable it as a test

---

## Monitoring Commands (Run in Separate Terminals)

```bash
# Real-time GPU monitoring (catches memory spikes before lockup)
nvidia-smi dmon -s umct -d 1

# System memory (unified pool — watch for swap usage)
watch -n 1 free -h

# Continuous GPU watch
watch -n 1 nvidia-smi

# Profile CUDA kernels (if investigating specific bottlenecks)
nsys profile --trace=cuda,nvtx python train.py
```

---

## Slow Training (If Stable But < 500 tokens/sec)

```python
# 1. Verify TF32 is enabled
torch.backends.cuda.matmul.allow_tf32 = True

# 2. Use compiled mode (PyTorch 2.0+)
model = torch.compile(model)

# 3. Optimize data loading
# dataloader_num_workers: 4
# dataloader_pin_memory: true
```

---

## Poor Convergence (Loss Not Decreasing)

```yaml
learning_rate: 0.0001        # Halve from 0.0002
warmup_steps: 200            # Double from 100
gradient_accumulation_steps: 16  # Larger effective batch = smoother gradients
lora_alpha: 16               # Reduce if model is unstable (default is 2x rank)
```

---

## QLoRA Memory Math (For Calculating Custom Configs)

- **Model weights**: `num_params × 0.5 bytes` (4-bit quantized)
- **LoRA adapters**: `trainable_params × 2 bytes` (FP16)
- **Optimizer (AdamW)**: `trainable_params × 2 states × 4 bytes`
- **Gradients**: `trainable_params × 2 bytes`
- **Activations (no checkpointing)**: `batch × seq_len × hidden_dim × num_layers × 4 bytes`
- **Activations (with checkpointing)**: ~3-4x less than above

For Llama 3.1 70B: hidden_dim=8192, num_layers=80

**Without checkpointing** (seq_len=2048, batch=1):
`1 × 2048 × 8192 × 80 × 4 = ~5.2GB`

**With checkpointing** (every 8 layers):
`~1.5GB per sample`

---

## Adapter Merging (Post-Training)

**Do NOT merge while quantized.** Load the base model in FP16/BF16 first:

```python
# Load base model in FULL precision for merging
model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16,  # NOT 4-bit
    device_map="auto"
)

# Then load and merge adapters
model = PeftModel.from_pretrained(model, adapter_path)
model = model.merge_and_unload()
model = model.to(torch.bfloat16)
```

---

## Quick Deployment After Training

```bash
# Convert to GGUF for llama.cpp inference
python convert.py ./outputs/merged-model/ \
    --outfile ./outputs/model.gguf --outtype f16

# Quantize to Q5_K_M (good quality/size balance)
./quantize ./outputs/model.gguf ./outputs/model-Q5_K_M.gguf Q5_K_M

# Serve with vLLM
vllm serve ./outputs/merged-model \
    --dtype bfloat16 --max-model-len 8192
```
