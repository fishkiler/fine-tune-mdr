#!/usr/bin/env python3
"""
============================================================================
Throughput Benchmark — Gradient Checkpointing / Batch Size / Sequence Length
============================================================================
Runs short 10-step training bursts with different configurations and reports
tokens/sec for each.  Streams live CPU, memory, GPU, and training metrics
to the dashboard at /bench.

Designed for DGX Spark GB10 (128 GB unified memory).

Usage:
    python bench_throughput.py
    python bench_throughput.py --config config.yaml
    python bench_throughput.py --steps 20        # more steps per trial
    python bench_throughput.py --dashboard-url http://localhost:28000
============================================================================
"""

import argparse
import gc
import logging
import os
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import httpx
import psutil
import torch
import yaml
from datasets import Dataset, load_from_disk
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Dashboard client ──────────────────────────────────────────────────────

class DashboardClient:
    """Non-blocking HTTP client that posts benchmark metrics to the dashboard."""

    def __init__(self, url: str | None):
        self.url = url.rstrip("/") if url else None
        self.client = httpx.Client(timeout=1.0) if url else None

    def post(self, endpoint: str, data: dict):
        if not self.client:
            return
        try:
            self.client.post(f"{self.url}{endpoint}", json=data)
        except Exception:
            pass  # never crash the benchmark

    def close(self):
        if self.client:
            self.client.close()


# ── Benchmark callback ────────────────────────────────────────────────────

class MemoryCeilingExceeded(Exception):
    """Raised when system memory usage exceeds the safety ceiling."""
    pass


class BenchmarkCallback(TrainerCallback):
    """Tracks per-step timing, system metrics, and training metrics.
    Posts everything to the dashboard in real time.

    On DGX Spark unified memory, PyTorch's OutOfMemoryError never fires —
    the CUDA allocator maps into system RAM until the kernel OOM-kills
    everything.  We enforce a soft ceiling here to abort the trial before
    the system locks up."""

    def __init__(self, trial_label: str, tokens_per_step: int, dashboard: DashboardClient,
                 max_system_mem_pct: float = 85.0):
        self.trial_label = trial_label
        self.tokens_per_step = tokens_per_step
        self.dashboard = dashboard
        self.max_system_mem_pct = max_system_mem_pct
        self.step_times: list[float] = []
        self.step_metrics: list[dict] = []
        self._step_start: float = 0.0
        self._process = psutil.Process()

    def _collect_system_metrics(self) -> dict:
        """Collect CPU, RAM, and GPU metrics."""
        # CPU — process-level
        cpu_percent = self._process.cpu_percent()

        # RAM
        vm = psutil.virtual_memory()
        proc_mem = self._process.memory_info()

        # GPU
        gpu_allocated = torch.cuda.memory_allocated() / (1024**3)
        gpu_reserved = torch.cuda.memory_reserved() / (1024**3)
        gpu_peak = torch.cuda.max_memory_allocated() / (1024**3)
        gpu_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        return {
            "cpu_percent": round(cpu_percent, 1),
            "ram_used_gb": round(proc_mem.rss / (1024**3), 2),
            "ram_total_gb": round(vm.total / (1024**3), 2),
            "ram_system_percent": round(vm.percent, 1),
            "gpu_allocated_gb": round(gpu_allocated, 2),
            "gpu_reserved_gb": round(gpu_reserved, 2),
            "gpu_peak_gb": round(gpu_peak, 2),
            "gpu_total_gb": round(gpu_total, 2),
            "gpu_percent": round(gpu_allocated / gpu_total * 100, 1) if gpu_total > 0 else 0,
        }

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_start = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        elapsed = time.perf_counter() - self._step_start
        self.step_times.append(elapsed)

        # ── Memory ceiling check (unified memory OOM prevention) ──────
        mem_pct = psutil.virtual_memory().percent
        if mem_pct > self.max_system_mem_pct:
            log.warning(
                f"  MEMORY CEILING: system at {mem_pct:.1f}% "
                f"(limit {self.max_system_mem_pct}%) — aborting trial"
            )
            raise MemoryCeilingExceeded(
                f"System memory at {mem_pct:.1f}% exceeds "
                f"{self.max_system_mem_pct}% ceiling"
            )

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return

        step_time = self.step_times[-1] if self.step_times else 0
        tokens_per_sec = self.tokens_per_step / step_time if step_time > 0 else 0

        sys_metrics = self._collect_system_metrics()

        metric = {
            "event": "bench_step",
            "trial": self.trial_label,
            "step": state.global_step,
            "max_steps": state.max_steps,
            "loss": logs.get("loss"),
            "learning_rate": logs.get("learning_rate"),
            "grad_norm": logs.get("grad_norm"),
            "step_time": round(step_time, 4),
            "tokens_per_sec": round(tokens_per_sec, 1),
            "tokens_per_step": self.tokens_per_step,
            "timestamp": time.time(),
            **sys_metrics,
        }

        self.step_metrics.append(metric)
        self.dashboard.post("/bench/log", metric)


# ── Trial configuration ──────────────────────────────────────────────────

@dataclass
class TrialConfig:
    label: str
    batch_size: int
    seq_length: int
    grad_ckpt: bool
    grad_accum: int = 1


# ── Helpers ───────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def format_dataset(ds: Dataset, tokenizer) -> Dataset:
    columns = ds.column_names
    if "text" in columns:
        return ds
    if "messages" not in columns:
        raise ValueError(f"Dataset needs 'text' or 'messages' column, got: {columns}")
    return ds.map(
        lambda ex: {
            "text": tokenizer.apply_chat_template(
                ex["messages"], tokenize=False, add_generation_prompt=False
            )
        },
        remove_columns=[c for c in columns if c != "text"],
        desc="Formatting",
    )


def cleanup():
    """Free GPU memory between trials — aggressive to prevent fragmentation."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    gc.collect()  # second pass catches ref cycles freed by first pass


def run_trial(
    trial: TrialConfig,
    cfg: dict,
    tokenizer,
    train_ds: Dataset,
    num_steps: int,
    dashboard: DashboardClient,
    max_mem_pct: float = 85.0,
) -> dict:
    """Run a single benchmark trial and return results."""
    log.info("=" * 70)
    log.info(f"TRIAL: {trial.label}")
    log.info(
        f"  batch_size={trial.batch_size}  seq_length={trial.seq_length}  "
        f"grad_ckpt={trial.grad_ckpt}  grad_accum={trial.grad_accum}"
    )
    log.info("=" * 70)

    cleanup()

    # Notify dashboard of trial start
    dashboard.post("/bench/log", {
        "event": "trial_start",
        "trial": trial.label,
        "batch_size": trial.batch_size,
        "seq_length": trial.seq_length,
        "grad_ckpt": trial.grad_ckpt,
        "grad_accum": trial.grad_accum,
        "num_steps": num_steps,
        "timestamp": time.time(),
    })

    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]

    # ── Load model ────────────────────────────────────────────────────
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"],
        torch_dtype=torch.bfloat16,
        attn_implementation=model_cfg.get("attn_implementation", "sdpa"),
    )
    model = model.to("cuda")

    # ── Gradient checkpointing ────────────────────────────────────────
    if trial.grad_ckpt:
        model.gradient_checkpointing_enable()

    # ── LoRA ──────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=TaskType.CAUSAL_LM,
        use_rslora=lora_cfg.get("use_rslora", False),
    )
    model = get_peft_model(model, lora_config)

    # ── Tokens per step ───────────────────────────────────────────────
    tokens_per_step = trial.batch_size * trial.grad_accum * trial.seq_length
    bench_cb = BenchmarkCallback(trial.label, tokens_per_step, dashboard, max_mem_pct)

    # ── SFTConfig ─────────────────────────────────────────────────────
    bench_dir = f"outputs/bench/{trial.label.replace(' ', '_')}"
    os.makedirs(bench_dir, exist_ok=True)

    sft_config = SFTConfig(
        output_dir=bench_dir,
        max_steps=num_steps,
        per_device_train_batch_size=trial.batch_size,
        gradient_accumulation_steps=trial.grad_accum,
        learning_rate=2e-4,
        warmup_steps=2,
        lr_scheduler_type="constant",
        optim="adamw_torch",
        bf16=True,
        fp16=False,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        dataset_text_field="text",
        max_length=trial.seq_length,
        packing=True,
        dataloader_num_workers=0,
        seed=42,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_ds,
        args=sft_config,
        callbacks=[bench_cb],
    )

    # ── Run ───────────────────────────────────────────────────────────
    oom = False
    try:
        wall_start = time.perf_counter()
        trainer.train()
        wall_total = time.perf_counter() - wall_start
    except torch.cuda.OutOfMemoryError:
        log.warning(f"  OOM on trial: {trial.label}")
        oom = True
        wall_total = 0.0
    except MemoryCeilingExceeded:
        log.warning(f"  OOM (memory ceiling) on trial: {trial.label}")
        oom = True
        wall_total = 0.0
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            log.warning(f"  OOM on trial: {trial.label}")
            oom = True
            wall_total = 0.0
        else:
            raise

    # ── Results ───────────────────────────────────────────────────────
    peak_mem_gb = torch.cuda.max_memory_allocated() / (1024**3)

    # Drop first 2 steps (warmup) for throughput calculation
    warmup_skip = 2
    step_times = bench_cb.step_times

    if oom or len(step_times) <= warmup_skip:
        result = {
            "label": trial.label,
            "batch_size": trial.batch_size,
            "seq_length": trial.seq_length,
            "grad_ckpt": trial.grad_ckpt,
            "grad_accum": trial.grad_accum,
            "tokens_per_step": tokens_per_step,
            "steps_completed": len(step_times),
            "oom": True,
            "tokens_per_sec": 0.0,
            "sec_per_step": 0.0,
            "peak_mem_gb": round(peak_mem_gb, 2),
            "wall_total": round(wall_total, 2),
        }
    else:
        steady_times = step_times[warmup_skip:]
        avg_step_time = sum(steady_times) / len(steady_times)
        tokens_per_sec = tokens_per_step / avg_step_time

        result = {
            "label": trial.label,
            "batch_size": trial.batch_size,
            "seq_length": trial.seq_length,
            "grad_ckpt": trial.grad_ckpt,
            "grad_accum": trial.grad_accum,
            "tokens_per_step": tokens_per_step,
            "steps_completed": len(step_times),
            "oom": False,
            "tokens_per_sec": round(tokens_per_sec, 1),
            "sec_per_step": round(avg_step_time, 4),
            "peak_mem_gb": round(peak_mem_gb, 2),
            "wall_total": round(wall_total, 2),
        }

    # Notify dashboard of trial end
    dashboard.post("/bench/log", {
        "event": "trial_end",
        "timestamp": time.time(),
        **result,
    })

    # ── Cleanup model ─────────────────────────────────────────────────
    del trainer, model
    cleanup()

    return result


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Throughput benchmark for DGX Spark")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--steps", type=int, default=10, help="Steps per trial")
    parser.add_argument("--dashboard-url", default=None,
                        help="Dashboard URL (default: from config dashboard.port)")
    parser.add_argument("--min-free-gb", type=float, default=15.0,
                        help="Skip trial if available RAM < this many GB (default: 15)")
    parser.add_argument("--resume-from", type=int, default=1, metavar="N",
                        help="Resume from trial N (1-indexed, skips trials 1..N-1)")
    parser.add_argument("--max-mem-pct", type=float, default=85.0,
                        help="Abort trial if system memory exceeds this %% (default: 85)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # ── Dashboard connection ──────────────────────────────────────────
    dash_url = args.dashboard_url
    if not dash_url:
        port = cfg.get("dashboard", {}).get("port", 28000)
        dash_url = f"http://localhost:{port}"
    dashboard = DashboardClient(dash_url)
    log.info(f"Dashboard: {dash_url}/bench")

    # ── CUDA memory allocator — prevent fragmentation-induced lockups ─
    # On DGX Spark unified memory, large cached blocks cause driver-level
    # semaphore holds during defrag that freeze the entire system.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    log.info(f"PYTORCH_CUDA_ALLOC_CONF={os.environ['PYTORCH_CUDA_ALLOC_CONF']}")

    # ── TF32 ──────────────────────────────────────────────────────────
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # ── Tokenizer ─────────────────────────────────────────────────────
    log.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Dataset ───────────────────────────────────────────────────────
    local_dir = Path(cfg["dataset"]["local_dir"])
    if (local_dir / "train").exists():
        log.info(f"Loading dataset from {local_dir}/train...")
        train_ds = load_from_disk(str(local_dir / "train"))
    else:
        log.error(f"No prepared dataset at {local_dir}/train — run data prep first.")
        sys.exit(1)
    train_ds = format_dataset(train_ds, tokenizer)
    log.info(f"Dataset: {len(train_ds)} examples")

    # ── Define trials (ordered safest → riskiest) ──────────────────
    trials = [
        # Phase A: Baseline — current production config (low risk)
        TrialConfig("baseline_b8_s1024",          batch_size=8,  seq_length=1024, grad_ckpt=False),
        TrialConfig("gc_b8_s1024",                batch_size=8,  seq_length=1024, grad_ckpt=True),

        # Phase B: Moderate scale-up — 2x batch or short seq (medium risk)
        TrialConfig("b16_s1024",                  batch_size=16, seq_length=1024, grad_ckpt=False),
        TrialConfig("gc_b16_s1024",               batch_size=16, seq_length=1024, grad_ckpt=True),
        TrialConfig("gc_b32_s512",                batch_size=32, seq_length=512,  grad_ckpt=True),

        # Phase C: Longer sequences — same batch, 2x seq (medium risk)
        TrialConfig("b8_s2048",                   batch_size=8,  seq_length=2048, grad_ckpt=False),
        TrialConfig("gc_b8_s2048",                batch_size=8,  seq_length=2048, grad_ckpt=True),

        # Phase D: Larger batches — 4–8x batch (med-high risk)
        TrialConfig("b32_s1024",                  batch_size=32, seq_length=1024, grad_ckpt=False),
        TrialConfig("gc_b32_s1024",               batch_size=32, seq_length=1024, grad_ckpt=True),
        TrialConfig("gc_b64_s512",                batch_size=64, seq_length=512,  grad_ckpt=True),

        # Phase E: Aggressive — largest memory footprints (high risk)
        TrialConfig("gc_b16_s2048",               batch_size=16, seq_length=2048, grad_ckpt=True),
        TrialConfig("gc_b64_s1024",               batch_size=64, seq_length=1024, grad_ckpt=True),
    ]

    # Notify dashboard of benchmark start
    trial_list = [asdict(t) for t in trials]
    dashboard.post("/bench/log", {
        "event": "bench_start",
        "total_trials": len(trials),
        "steps_per_trial": args.steps,
        "trials": trial_list,
        "timestamp": time.time(),
    })

    # ── Run trials ────────────────────────────────────────────────────
    resume_idx = args.resume_from - 1  # convert to 0-indexed
    if resume_idx > 0:
        log.info(f"Resuming from trial {args.resume_from}/{len(trials)} — skipping first {resume_idx} trials")

    results: list[dict] = []
    for i, trial in enumerate(trials):
        if i < resume_idx:
            log.info(f"\n>>> Trial {i + 1}/{len(trials)}: {trial.label}  [SKIPPED — already completed]")
            continue
        log.info(f"\n>>> Trial {i + 1}/{len(trials)}: {trial.label}")

        # Pre-flight memory safety check
        available_gb = psutil.virtual_memory().available / (1024**3)
        if available_gb < args.min_free_gb:
            log.warning(
                f"Skipping {trial.label} — only {available_gb:.1f} GB free "
                f"(need {args.min_free_gb:.0f} GB)"
            )
            tokens_per_step = trial.batch_size * trial.grad_accum * trial.seq_length
            skip_result = {
                "label": trial.label,
                "batch_size": trial.batch_size,
                "seq_length": trial.seq_length,
                "grad_ckpt": trial.grad_ckpt,
                "grad_accum": trial.grad_accum,
                "tokens_per_step": tokens_per_step,
                "steps_completed": 0,
                "oom": False,
                "skipped": True,
                "skip_reason": f"Only {available_gb:.1f} GB free (need {args.min_free_gb:.0f} GB)",
                "tokens_per_sec": 0.0,
                "sec_per_step": 0.0,
                "peak_mem_gb": 0.0,
                "wall_total": 0.0,
            }
            dashboard.post("/bench/log", {
                "event": "trial_skip",
                "trial": trial.label,
                "available_gb": round(available_gb, 1),
                "min_free_gb": args.min_free_gb,
                "reason": skip_result["skip_reason"],
                "timestamp": time.time(),
                **skip_result,
            })
            results.append(skip_result)
            log.info(f"  RESULT: SKIP  |  {available_gb:.1f} GB available")
            continue

        result = run_trial(trial, cfg, tokenizer, train_ds, args.steps, dashboard, args.max_mem_pct)
        results.append(result)

        if result["oom"]:
            log.info(f"  RESULT: OOM  |  peak mem: {result['peak_mem_gb']:.1f} GB")
        else:
            log.info(
                f"  RESULT: {result['tokens_per_sec']:,.0f} tok/s  |  "
                f"{result['sec_per_step']:.2f} s/step  |  "
                f"peak mem: {result['peak_mem_gb']:.1f} GB"
            )

    # Notify dashboard of benchmark end
    dashboard.post("/bench/log", {
        "event": "bench_end",
        "results": results,
        "timestamp": time.time(),
    })

    # ── Summary table ─────────────────────────────────────────────────
    print("\n")
    print("=" * 100)
    print("BENCHMARK RESULTS")
    print("=" * 100)
    header = (
        f"{'Trial':<25} {'Batch':>5} {'SeqLen':>6} {'GC':>4} "
        f"{'Tok/Step':>10} {'Tok/s':>10} {'s/step':>8} "
        f"{'PeakMem':>8} {'Status':>8}"
    )
    print(header)
    print("-" * 100)

    best_tps = 0.0
    best_label = ""

    for r in results:
        skipped = r.get("skipped", False)
        if skipped:
            status = "SKIP"
        elif r["oom"]:
            status = "OOM"
        else:
            status = "OK"

        ok = not r["oom"] and not skipped
        tps = f"{r['tokens_per_sec']:>10,.0f}" if ok else f"{'—':>10}"
        sps = f"{r['sec_per_step']:>8.2f}" if ok else f"{'—':>8}"
        mem = f"{r['peak_mem_gb']:>7.1f}G" if not skipped else f"{'—':>7} "

        print(
            f"{r['label']:<25} {r['batch_size']:>5} {r['seq_length']:>6} "
            f"{'Y' if r['grad_ckpt'] else 'N':>4} "
            f"{r['tokens_per_step']:>10,} {tps} {sps} "
            f"{mem} {status:>8}"
        )

        if ok and r["tokens_per_sec"] > best_tps:
            best_tps = r["tokens_per_sec"]
            best_label = r["label"]

    print("-" * 100)
    if best_label:
        print(f"\nBEST: {best_label}  —  {best_tps:,.0f} tokens/sec")

        best = next(r for r in results if r["label"] == best_label)
        print(f"\nRecommended config.yaml updates:")
        print(f"  per_device_train_batch_size: {best['batch_size']}")
        print(f"  max_seq_length: {best['seq_length']}")
        print(f"  use_gradient_checkpointing: {str(best['grad_ckpt']).lower()}")
    else:
        print("\nAll trials OOM — try smaller batch sizes or shorter sequences.")

    print()
    dashboard.close()


if __name__ == "__main__":
    main()
