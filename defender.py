#!/usr/bin/env python3
"""
============================================================================
Atari Defender — RL Training with Stable-Baselines3 PPO
============================================================================
Single-file CLI for training and playing Defender using PPO with CNN policy.
Streams live metrics to the dashboard server via httpx POST.

Usage:
    python defender.py train                    # Train PPO for 1M steps
    python defender.py train --steps 2000000    # Custom step count
    python defender.py play                     # Load best model, render live
    python defender.py play --model path/to/model.zip
============================================================================
"""

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path

import ale_py
import gymnasium
import httpx
import numpy as np
import torch
import yaml

# Register ALE Atari environments with gymnasium
gymnasium.register_envs(ale_py)

from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack, VecTransposeImage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


# ============================================================================
# Config
# ============================================================================

def load_config() -> dict:
    """Load RL training config from config.yaml."""
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("rl_training", {}).get("defender", {})


# ============================================================================
# Dashboard Callback
# ============================================================================

class RLDashboardCallback(BaseCallback):
    """Posts RL training metrics to the dashboard SSE server."""

    def __init__(
        self,
        dashboard_url: str = "http://localhost:28000",
        log_path: Path | None = None,
        total_timesteps: int = 1_000_000,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.dashboard_url = dashboard_url.rstrip("/")
        self.client = httpx.Client(timeout=0.5)
        self.log_path = log_path
        self.total_timesteps = total_timesteps
        self.start_time: float | None = None
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.best_mean_reward = -np.inf
        self.last_post_time = 0.0
        self.post_interval = 2.0  # seconds between dashboard posts

    def _post(self, data: dict):
        """POST to dashboard server, silently ignore failures."""
        try:
            self.client.post(f"{self.dashboard_url}/log", json=self._sanitize(data))
        except Exception:
            pass

    @staticmethod
    def _sanitize(data: dict) -> dict:
        """Convert numpy/torch scalars to native Python types for JSON."""
        out = {}
        for k, v in data.items():
            if isinstance(v, (np.floating, np.integer)):
                out[k] = v.item()
            elif isinstance(v, np.ndarray):
                out[k] = v.tolist()
            else:
                out[k] = v
        return out

    def _log_jsonl(self, data: dict):
        """Append a line to the training log JSONL file."""
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(self._sanitize(data)) + "\n")

    def _gpu_metrics(self) -> dict:
        """Collect GPU memory and temperature."""
        metrics = {}
        if not torch.cuda.is_available():
            return metrics
        metrics["gpu_memory_allocated_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
        metrics["gpu_memory_reserved_gb"] = round(torch.cuda.memory_reserved() / 1e9, 2)
        metrics["gpu_memory_total_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / 1e9, 2
        )
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            metrics["gpu_temp_c"] = pynvml.nvmlDeviceGetTemperature(
                handle, pynvml.NVML_TEMPERATURE_GPU
            )
        except Exception:
            pass
        try:
            cpu_temps = []
            for zone in sorted(Path("/sys/class/thermal").glob("thermal_zone*")):
                temp_file = zone / "temp"
                if temp_file.exists():
                    cpu_temps.append(int(temp_file.read_text().strip()) / 1000)
            if cpu_temps:
                metrics["cpu_temp_c"] = round(max(cpu_temps), 1)
        except Exception:
            pass
        return metrics

    def _on_training_start(self):
        self.start_time = time.time()
        self._post({
            "event": "rl_train_begin",
            "env_id": "ALE/Defender-v5",
            "game_name": "Defender",
            "algorithm": "PPO",
            "total_timesteps": self.total_timesteps,
            "timestamp": self.start_time,
        })
        log.info("Dashboard callback initialized — streaming to %s", self.dashboard_url)

    def _on_step(self) -> bool:
        # Collect episode info from monitor wrappers
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                self.episode_rewards.append(ep["r"])
                self.episode_lengths.append(ep["l"])

        # Throttle dashboard posts
        now = time.time()
        if now - self.last_post_time < self.post_interval:
            return True
        self.last_post_time = now

        elapsed = now - self.start_time if self.start_time else 0
        timestep = self.num_timesteps
        fps = timestep / elapsed if elapsed > 0 else 0

        # ETA
        remaining = self.total_timesteps - timestep
        eta_seconds = remaining / fps if fps > 0 else 0

        # Episode stats (rolling window of 100)
        recent_rewards = self.episode_rewards[-100:] if self.episode_rewards else []
        recent_lengths = self.episode_lengths[-100:] if self.episode_lengths else []
        mean_reward = float(np.mean(recent_rewards)) if recent_rewards else 0.0
        best_reward = float(np.max(recent_rewards)) if recent_rewards else 0.0
        mean_length = float(np.mean(recent_lengths)) if recent_lengths else 0.0

        # Track best mean reward
        if mean_reward > self.best_mean_reward and len(recent_rewards) >= 10:
            self.best_mean_reward = mean_reward

        # Get training losses from logger
        metrics = {
            "event": "rl_metrics",
            "timestep": timestep,
            "total_timesteps": self.total_timesteps,
            "episodes_completed": len(self.episode_rewards),
            "mean_reward": round(mean_reward, 2),
            "best_reward": round(best_reward, 2),
            "best_mean_reward": round(self.best_mean_reward, 2),
            "mean_episode_length": round(mean_length, 1),
            "fps": round(fps, 1),
            "elapsed_seconds": round(elapsed, 1),
            "eta_seconds": round(eta_seconds, 0),
            "timestamp": now,
        }

        # Add SB3 logger values if available
        if hasattr(self, "logger") and self.logger is not None:
            name_to_value = getattr(self.logger, "name_to_value", {})
            if name_to_value:
                metrics["value_loss"] = name_to_value.get("train/value_loss")
                metrics["policy_loss"] = name_to_value.get("train/policy_gradient_loss")
                metrics["entropy"] = name_to_value.get("train/entropy_loss")
                metrics["approx_kl"] = name_to_value.get("train/approx_kl")
                metrics["clip_fraction"] = name_to_value.get("train/clip_fraction")
                metrics["explained_variance"] = name_to_value.get("train/explained_variance")

        metrics.update(self._gpu_metrics())

        # Clean None values
        metrics = {k: v for k, v in metrics.items() if v is not None}

        self._post(metrics)
        self._log_jsonl(metrics)

        return True

    def _on_training_end(self):
        elapsed = time.time() - self.start_time if self.start_time else 0
        self._post({
            "event": "rl_train_end",
            "total_timesteps": self.num_timesteps,
            "episodes_completed": len(self.episode_rewards),
            "best_mean_reward": round(self.best_mean_reward, 2),
            "elapsed_seconds": round(elapsed, 1),
            "timestamp": time.time(),
        })
        self.client.close()
        log.info("Training complete — %d timesteps, best mean reward: %.2f",
                 self.num_timesteps, self.best_mean_reward)


# ============================================================================
# Best Model Callback
# ============================================================================

class SaveBestModelCallback(BaseCallback):
    """Saves the model when mean reward improves."""

    def __init__(self, save_path: Path, check_freq: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.check_freq = check_freq
        self.best_mean_reward = -np.inf
        self.episode_rewards: list[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                self.episode_rewards.append(ep["r"])

        if self.n_calls % self.check_freq == 0 and len(self.episode_rewards) >= 10:
            recent = self.episode_rewards[-100:]
            mean_reward = float(np.mean(recent))
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.save_path.mkdir(parents=True, exist_ok=True)
                path = self.save_path / "best_model"
                self.model.save(str(path))
                log.info("New best mean reward: %.2f — saved to %s", mean_reward, path)
        return True


# ============================================================================
# Train
# ============================================================================

def train(args):
    """Train PPO on Defender."""
    cfg = load_config()

    env_id = cfg.get("env_id", "ALE/Defender-v5")
    n_envs = cfg.get("n_envs", 8)
    total_timesteps = args.steps or cfg.get("total_timesteps", 1_000_000)
    output_dir = Path(cfg.get("output_dir", "adapters/games/defender"))
    checkpoint_dir = output_dir / "checkpoints"
    best_model_dir = output_dir / "best_model"
    log_path = output_dir / "training_log.jsonl"
    save_freq = cfg.get("save_freq", 100_000)
    dashboard_port = 28000

    # Load dashboard port from config
    with open(CONFIG_PATH) as f:
        full_cfg = yaml.safe_load(f)
    dashboard_port = full_cfg.get("dashboard", {}).get("port", 28000)
    dashboard_url = f"http://localhost:{dashboard_port}"

    # Create directories
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_model_dir.mkdir(parents=True, exist_ok=True)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.cuda.set_per_process_memory_fraction(0.85)
        log.info("CUDA available — using GPU with 85%% memory cap")
    else:
        log.info("CUDA not available — using CPU (training will be slow)")

    # Environment
    log.info("Creating %d parallel %s environments...", n_envs, env_id)
    env = make_atari_env(env_id, n_envs=n_envs, seed=42)
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)

    # PPO hyperparameters from config
    ppo_kwargs = {
        "learning_rate": cfg.get("learning_rate", 2.5e-4),
        "n_steps": cfg.get("n_steps", 128),
        "batch_size": cfg.get("batch_size", 256),
        "n_epochs": cfg.get("n_epochs", 4),
        "gamma": cfg.get("gamma", 0.99),
        "gae_lambda": cfg.get("gae_lambda", 0.95),
        "clip_range": cfg.get("clip_range", 0.1),
        "ent_coef": cfg.get("ent_coef", 0.01),
        "vf_coef": cfg.get("vf_coef", 0.5),
        "max_grad_norm": cfg.get("max_grad_norm", 0.5),
    }

    log.info("Initializing PPO with CnnPolicy...")
    log.info("Hyperparameters: %s", json.dumps(ppo_kwargs, indent=2))

    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        device=device,
        tensorboard_log=str(output_dir / "tb_logs"),
        **ppo_kwargs,
    )

    # Callbacks
    dashboard_cb = RLDashboardCallback(
        dashboard_url=dashboard_url,
        log_path=log_path,
        total_timesteps=total_timesteps,
    )

    # Checkpoint every save_freq steps (adjusted for n_envs)
    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, save_freq // n_envs),
        save_path=str(checkpoint_dir),
        name_prefix="defender_ppo",
    )

    best_model_cb = SaveBestModelCallback(
        save_path=best_model_dir,
        check_freq=max(1, 10_000 // n_envs),
    )

    # Cooperative SIGINT shutdown
    shutdown_requested = False
    original_sigint = signal.getsignal(signal.SIGINT)

    def sigint_handler(signum, frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            log.warning("Second SIGINT — forcing exit")
            signal.signal(signal.SIGINT, original_sigint)
            raise KeyboardInterrupt
        shutdown_requested = True
        log.info("SIGINT received — finishing current rollout, then saving...")

    signal.signal(signal.SIGINT, sigint_handler)

    # Train
    log.info("Starting training: %s timesteps on %s", f"{total_timesteps:,}", env_id)
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[dashboard_cb, checkpoint_cb, best_model_cb],
            progress_bar=True,
        )
    except KeyboardInterrupt:
        log.info("Training interrupted — saving final checkpoint...")

    # Save final model
    final_path = output_dir / "final_model"
    model.save(str(final_path))
    log.info("Final model saved to %s", final_path)

    env.close()
    log.info("Done.")


# ============================================================================
# Play
# ============================================================================

def play(args):
    """Load trained model and play Defender with rendering."""
    cfg = load_config()
    env_id = cfg.get("env_id", "ALE/Defender-v5")
    output_dir = Path(cfg.get("output_dir", "adapters/games/defender"))

    # Find model
    if args.model:
        model_path = args.model
    else:
        best = output_dir / "best_model" / "best_model.zip"
        final = output_dir / "final_model.zip"
        if best.exists():
            model_path = str(best)
        elif final.exists():
            model_path = str(final)
        else:
            log.error("No model found. Train first with: python defender.py train")
            sys.exit(1)

    log.info("Loading model from %s", model_path)

    # Use same vectorized env pipeline as training (n_envs=1) for correct obs shape
    env = make_atari_env(env_id, n_envs=1, seed=42, env_kwargs={"render_mode": "human"})
    env = VecFrameStack(env, n_stack=4)
    env = VecTransposeImage(env)

    model = PPO.load(model_path)

    log.info("Playing %s — press Ctrl+C to stop", env_id)
    episode = 0
    try:
        obs = env.reset()
        total_reward = 0.0
        steps = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total_reward += reward[0]
            steps += 1
            if done[0]:
                episode += 1
                ep_info = info[0].get("episode", {})
                score = ep_info.get("r", total_reward)
                length = ep_info.get("l", steps)
                log.info("Episode %d: reward=%.0f, steps=%d", episode, score, length)
                total_reward = 0.0
                steps = 0
    except KeyboardInterrupt:
        log.info("Stopped after %d episodes", episode)
    finally:
        env.close()


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Atari Defender RL Training with PPO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # Train
    train_parser = sub.add_parser("train", help="Train PPO agent on Defender")
    train_parser.add_argument("--steps", type=int, default=None,
                              help="Total timesteps (overrides config)")

    # Play
    play_parser = sub.add_parser("play", help="Play Defender with trained model")
    play_parser.add_argument("--model", type=str, default=None,
                             help="Path to model .zip file")

    args = parser.parse_args()

    if args.command == "train":
        train(args)
    elif args.command == "play":
        play(args)


if __name__ == "__main__":
    main()
