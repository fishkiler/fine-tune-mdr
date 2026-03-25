# Colab Migration Log

## 2026-02-17: Moved Training from DGX Spark to Colab A100

### Checkpoint Timeline
- **Checkpoint-4275**: Uploaded to shared drive (`/mnt/ai_projects/checkpoints/`) and Google Drive for Colab
- **Checkpoint-4377**: Final DGX checkpoint (saved on graceful shutdown after Colab migration)
- **Gap**: ~102 steps (4275→4377) trained on DGX but not transferred to Colab. These steps will be re-done on Colab — negligible cost (~17 min on A100).

### DGX Training Stats
- Started: Feb 16, 2026
- Stopped: Feb 17, 2026 ~20:16 UTC (SIGTERM graceful shutdown)
- Steps completed: 4377 / 17479 (~25%)
- Speed: ~34 sec/step

### Colab A100 Config Changes
- Batch size: 16 → 8 (40GB VRAM constraint)
- Grad accumulation: 2 → 4 (effective batch stays 32, total steps unchanged)
- Attention: sdpa → flash_attention_2
- Save steps: 50 → 25 (Colab disconnect resilience)
- Time limit: 23h auto-save

### Files
- `colab_resume_training.ipynb` — Colab notebook (resuming from checkpoint-4275)
- `upload_checkpoint.sh` — DGX → shared drive upload script
