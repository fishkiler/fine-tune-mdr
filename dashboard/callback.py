"""
============================================================================
Fine-Tune MDR — Dashboard Training Callback
============================================================================
HuggingFace TrainerCallback that posts metrics to the dashboard server.
Non-blocking with 0.5s timeout — never crashes training.
============================================================================
"""

import logging
import time
from pathlib import Path
from typing import Optional

import httpx
import torch
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

log = logging.getLogger(__name__)


class DashboardCallback(TrainerCallback):
    """Posts training metrics to the dashboard SSE server."""

    def __init__(self, dashboard_url: str = "http://localhost:8000", timeout: float = 0.5):
        self.dashboard_url = dashboard_url.rstrip("/")
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)
        self.train_start_time: Optional[float] = None
        self.step_times: list[float] = []

    # ----------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------

    def _post(self, endpoint: str, data: dict):
        """POST to dashboard server, silently ignore failures."""
        try:
            self.client.post(f"{self.dashboard_url}{endpoint}", json=data)
        except Exception:
            pass  # never crash training

    def _gpu_metrics(self) -> dict:
        """Collect GPU memory stats and CPU/GPU temperatures."""
        metrics = {}
        if not torch.cuda.is_available():
            return metrics

        metrics["gpu_memory_allocated_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
        metrics["gpu_memory_reserved_gb"] = round(torch.cuda.memory_reserved() / 1e9, 2)
        metrics["gpu_memory_total_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 2
        )

        # CPU temp — read from ACPI thermal zones
        try:
            cpu_temps = []
            for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
                temp_file = zone / "temp"
                if temp_file.exists():
                    cpu_temps.append(int(temp_file.read_text().strip()) / 1000)
            if cpu_temps:
                metrics["cpu_temp_c"] = round(max(cpu_temps), 1)
                metrics["cpu_temp_avg_c"] = round(sum(cpu_temps) / len(cpu_temps), 1)
        except Exception:
            pass

        # GPU temp — via pynvml
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            metrics["gpu_temp_c"] = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
        except Exception:
            pass

        return metrics

    # ----------------------------------------------------------------
    # Callbacks
    # ----------------------------------------------------------------

    def on_train_begin(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        self.train_start_time = time.time()
        self._post("/log", {
            "event": "train_begin",
            "max_steps": state.max_steps,
            "num_epochs": args.num_train_epochs,
            "batch_size": args.per_device_train_batch_size,
            "gradient_accumulation": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "timestamp": self.train_start_time,
        })

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: Optional[dict] = None,
        **kwargs,
    ):
        if logs is None:
            return

        now = time.time()
        elapsed = now - self.train_start_time if self.train_start_time else 0

        # Track step timing for ETA
        self.step_times.append(now)
        if len(self.step_times) > 100:
            self.step_times = self.step_times[-100:]

        # Calculate samples/sec and ETA
        samples_per_sec = 0.0
        eta_seconds = 0.0
        if len(self.step_times) >= 2:
            recent_duration = self.step_times[-1] - self.step_times[0]
            recent_steps = len(self.step_times) - 1
            if recent_duration > 0:
                steps_per_sec = recent_steps / recent_duration
                samples_per_sec = steps_per_sec * args.per_device_train_batch_size * args.gradient_accumulation_steps
                remaining_steps = state.max_steps - state.global_step
                eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0

        metrics = {
            "event": "metrics",
            "step": state.global_step,
            "epoch": round(state.epoch, 4) if state.epoch else 0,
            "max_steps": state.max_steps,
            "loss": logs.get("loss"),
            "learning_rate": logs.get("learning_rate"),
            "grad_norm": logs.get("grad_norm"),
            "elapsed_seconds": round(elapsed, 1),
            "samples_per_sec": round(samples_per_sec, 1),
            "eta_seconds": round(eta_seconds, 0),
            "timestamp": now,
        }
        metrics.update(self._gpu_metrics())

        self._post("/log", metrics)

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        elapsed = time.time() - self.train_start_time if self.train_start_time else 0
        self._post("/log", {
            "event": "train_end",
            "total_steps": state.global_step,
            "elapsed_seconds": round(elapsed, 1),
            "timestamp": time.time(),
        })
        self.client.close()

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[dict] = None,
        **kwargs,
    ):
        if metrics is None:
            return
        self._post("/log", {
            "event": "eval",
            "step": state.global_step,
            "eval_loss": metrics.get("eval_loss"),
            "timestamp": time.time(),
        })


class TimeLimitCallback(TrainerCallback):
    """Stops training after a wall-clock time limit, saving a proper checkpoint.

    Useful for nightly training sessions where you want automatic shutdown
    after N hours. Set time_limit_hours=0 to disable (default).
    """

    def __init__(self, time_limit_hours: float = 0):
        self.time_limit_seconds = time_limit_hours * 3600 if time_limit_hours > 0 else 0
        self.start_time: Optional[float] = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        if self.time_limit_seconds > 0:
            log.info(
                f"TimeLimitCallback: will stop after {self.time_limit_seconds / 3600:.1f} hours"
            )

    def on_step_end(self, args, state, control, **kwargs):
        if self.time_limit_seconds <= 0 or self.start_time is None:
            return
        elapsed = time.time() - self.start_time
        if elapsed >= self.time_limit_seconds:
            log.warning(
                f"TimeLimitCallback: {elapsed / 3600:.1f}h elapsed, stopping training..."
            )
            control.should_training_stop = True
            control.should_save = True
