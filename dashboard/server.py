"""
============================================================================
Fine-Tune MDR — Dashboard SSE Server
============================================================================
FastAPI server that receives metrics from the training callback and fans
them out to browser clients via Server-Sent Events (SSE).

Usage:
    uvicorn dashboard.server:app --host 0.0.0.0 --port 8000
============================================================================
"""

import asyncio
import json
import logging
import signal
import sys
import time
from collections import deque
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ============================================================================
# App
# ============================================================================

app = FastAPI(title="Fine-Tune MDR Dashboard")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Metrics ring buffer
BUFFER_SIZE = 10_000
metrics_history: deque[dict] = deque(maxlen=BUFFER_SIZE)

# SSE subscribers
subscribers: list[asyncio.Queue] = []

# Training state
train_state: dict = {
    "status": "waiting",  # waiting | training | complete | error
    "config": {},
    "latest": {},
    "start_time": None,
}

# Benchmark state
bench_state: dict = {
    "status": "idle",  # idle | running | complete
    "config": {},
    "trials": [],       # trial configs
    "results": [],      # completed trial results
    "step_log": deque(maxlen=50_000),  # all per-step metrics
    "current_trial": None,
}
bench_subscribers: list[asyncio.Queue] = []

# Refresh state
refresh_state: dict = {
    "status": "idle",  # idle | running | complete | error
    "logs": deque(maxlen=5000),
    "started_at": None,
    "finished_at": None,
    "error": None,
    "process": None,  # asyncio subprocess reference
}
refresh_subscribers: list[asyncio.Queue] = []

# Training process state (subprocess management, separate from train_state metrics)
train_process_state: dict = {
    "status": "idle",  # idle | running | stopping | complete | error
    "logs": deque(maxlen=5000),
    "started_at": None,
    "stopped_at": None,
    "error": None,
    "process": None,  # asyncio subprocess reference
}
train_log_subscribers: list[asyncio.Queue] = []


# ============================================================================
# SSE Fan-Out
# ============================================================================

async def broadcast(data: dict):
    """Push a metric event to all connected SSE subscribers."""
    message = json.dumps(data)
    dead = []
    for q in subscribers:
        try:
            q.put_nowait(message)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        subscribers.remove(q)


async def broadcast_bench(data: dict):
    """Push a benchmark event to all connected bench SSE subscribers."""
    message = json.dumps(data)
    dead = []
    for q in bench_subscribers:
        try:
            q.put_nowait(message)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        bench_subscribers.remove(q)


async def broadcast_refresh(line: str):
    """Push a refresh log line to all connected refresh SSE subscribers."""
    dead = []
    for q in refresh_subscribers:
        try:
            q.put_nowait(line)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        refresh_subscribers.remove(q)


async def broadcast_train_log(line: str):
    """Push a training log line to all connected train log SSE subscribers."""
    dead = []
    for q in train_log_subscribers:
        try:
            q.put_nowait(line)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        train_log_subscribers.remove(q)


# ============================================================================
# Routes — Metric Ingestion
# ============================================================================

@app.post("/log")
async def receive_log(request: Request):
    """Receive metrics from the DashboardCallback."""
    data = await request.json()
    event = data.get("event", "unknown")

    if event == "train_begin":
        train_state["status"] = "training"
        train_state["config"] = data
        train_state["start_time"] = data.get("timestamp", time.time())
        log.info("Training started — dashboard is live.")

    elif event == "train_end":
        train_state["status"] = "complete"
        log.info("Training complete.")

    elif event == "metrics":
        train_state["latest"] = data
        # Auto-recover status after server restart (train_begin was missed)
        if train_state["status"] == "waiting":
            train_state["status"] = "training"
            train_state["start_time"] = data.get("timestamp", time.time())
            log.info("Training detected from metrics — status recovered.")

    metrics_history.append(data)
    await broadcast(data)

    return {"ok": True}


# ============================================================================
# Routes — SSE Streaming
# ============================================================================

@app.get("/metrics")
async def stream_metrics(request: Request):
    """SSE endpoint — replays full history on connect, then streams live."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
    subscribers.append(queue)

    async def event_generator():
        # Send full history first so charts populate immediately
        for entry in list(metrics_history):
            yield {"data": json.dumps(entry)}

        # Then stream live
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield {"data": message}
                except asyncio.TimeoutError:
                    # Send keepalive
                    yield {"comment": "keepalive"}
        finally:
            if queue in subscribers:
                subscribers.remove(queue)

    return EventSourceResponse(event_generator())


# ============================================================================
# Routes — REST Endpoints
# ============================================================================

@app.get("/state")
async def get_state():
    """Latest training state snapshot (for page reload)."""
    return JSONResponse({
        "status": train_state["status"],
        "config": train_state["config"],
        "latest": train_state["latest"],
    })


@app.get("/history")
async def get_history():
    """Full metrics history (for chart reconstruction)."""
    return JSONResponse(list(metrics_history))


# ============================================================================
# Routes — Dataset Refresh
# ============================================================================

@app.get("/refresh/manifest")
async def get_manifest():
    """Return the current dataset manifest."""
    manifest_path = PROJECT_ROOT / "data" / "manifest.json"
    if manifest_path.exists():
        return JSONResponse(json.loads(manifest_path.read_text()))
    return JSONResponse({"error": "No manifest found"}, status_code=404)


@app.get("/refresh/status")
async def get_refresh_status():
    """Return the current refresh process status."""
    return JSONResponse({
        "status": refresh_state["status"],
        "started_at": refresh_state["started_at"],
        "finished_at": refresh_state["finished_at"],
        "error": refresh_state["error"],
        "log_lines": len(refresh_state["logs"]),
    })


@app.post("/refresh/start")
async def start_refresh(request: Request):
    """Start a dataset refresh as a background subprocess."""
    if refresh_state["status"] == "running":
        return JSONResponse(
            {"error": "Refresh already in progress"},
            status_code=409,
        )

    body = await request.json() if request.headers.get("content-type") else {}
    days_back = body.get("days_back")
    skip_build = body.get("skip_build", False)

    # Build command
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "refresh_data.py")]
    if days_back:
        cmd.extend(["--days-back", str(int(days_back))])
    if skip_build:
        cmd.append("--skip-build")

    # Reset state
    refresh_state["status"] = "running"
    refresh_state["logs"].clear()
    refresh_state["started_at"] = time.time()
    refresh_state["finished_at"] = None
    refresh_state["error"] = None

    log.info(f"Starting refresh: {' '.join(cmd)}")

    # Launch subprocess
    asyncio.create_task(_run_refresh(cmd))

    return JSONResponse({"ok": True, "command": " ".join(cmd)})


@app.post("/refresh/cancel")
async def cancel_refresh():
    """Cancel a running refresh."""
    proc = refresh_state.get("process")
    if refresh_state["status"] != "running" or proc is None:
        return JSONResponse({"error": "No refresh running"}, status_code=409)

    try:
        proc.terminate()
    except ProcessLookupError:
        pass

    refresh_state["status"] = "error"
    refresh_state["error"] = "Cancelled by user"
    refresh_state["finished_at"] = time.time()
    line = "--- Refresh cancelled by user ---"
    refresh_state["logs"].append(line)
    await broadcast_refresh(line)
    log.info("Refresh cancelled by user.")

    return JSONResponse({"ok": True})


async def _run_refresh(cmd: list[str]):
    """Run the refresh subprocess and stream output to SSE subscribers."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
        )
        refresh_state["process"] = proc

        async for raw_line in proc.stdout:
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            refresh_state["logs"].append(line)
            await broadcast_refresh(line)

        await proc.wait()
        refresh_state["process"] = None

        if proc.returncode == 0:
            refresh_state["status"] = "complete"
            line = "--- Refresh completed successfully ---"
        else:
            refresh_state["status"] = "error"
            refresh_state["error"] = f"Exit code {proc.returncode}"
            line = f"--- Refresh failed (exit code {proc.returncode}) ---"

        refresh_state["logs"].append(line)
        refresh_state["finished_at"] = time.time()
        await broadcast_refresh(line)
        log.info(f"Refresh finished: {refresh_state['status']}")

    except Exception as e:
        refresh_state["status"] = "error"
        refresh_state["error"] = str(e)
        refresh_state["finished_at"] = time.time()
        refresh_state["process"] = None
        line = f"--- Refresh error: {e} ---"
        refresh_state["logs"].append(line)
        await broadcast_refresh(line)
        log.error(f"Refresh error: {e}")


@app.get("/refresh/logs")
async def stream_refresh_logs(request: Request):
    """SSE endpoint — streams refresh subprocess output in real time."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=5000)
    refresh_subscribers.append(queue)

    async def event_generator():
        # Replay existing log lines first
        for line in list(refresh_state["logs"]):
            yield {"data": line}

        # Stream live
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    line = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield {"data": line}
                except asyncio.TimeoutError:
                    yield {"comment": "keepalive"}
                    # Stop streaming if refresh is done
                    if refresh_state["status"] in ("complete", "error", "idle"):
                        break
        finally:
            if queue in refresh_subscribers:
                refresh_subscribers.remove(queue)

    return EventSourceResponse(event_generator())


# ============================================================================
# Routes — Training Process Management
# ============================================================================

@app.get("/train/state")
async def get_train_process_state():
    """Return the current training process status."""
    return JSONResponse({
        "status": train_process_state["status"],
        "started_at": train_process_state["started_at"],
        "stopped_at": train_process_state["stopped_at"],
        "error": train_process_state["error"],
        "log_lines": len(train_process_state["logs"]),
    })


@app.post("/train/start")
async def start_training(request: Request):
    """Start training as a background subprocess."""
    if train_process_state["status"] == "running":
        return JSONResponse(
            {"error": "Training already in progress"},
            status_code=409,
        )

    body = await request.json() if request.headers.get("content-type") else {}
    fresh = body.get("fresh", False)

    # Build command
    cmd = ["bash", str(PROJECT_ROOT / "train.sh")]
    if fresh:
        cmd.append("--fresh")

    # Reset state
    train_process_state["status"] = "running"
    train_process_state["logs"].clear()
    train_process_state["started_at"] = time.time()
    train_process_state["stopped_at"] = None
    train_process_state["error"] = None

    # Also reset the metrics train_state so the dashboard picks up the new run
    train_state["status"] = "waiting"
    train_state["latest"] = {}

    log.info(f"Starting training: {' '.join(cmd)}")

    asyncio.create_task(_run_training(cmd))

    return JSONResponse({"ok": True, "command": " ".join(cmd)})


@app.post("/train/stop")
async def stop_training():
    """Send SIGINT to training process for cooperative shutdown."""
    proc = train_process_state.get("process")
    if train_process_state["status"] != "running" or proc is None:
        return JSONResponse({"error": "No training running"}, status_code=409)

    try:
        proc.send_signal(signal.SIGINT)
        train_process_state["status"] = "stopping"
        line = "--- Stop signal sent (saving checkpoint...) ---"
        train_process_state["logs"].append(line)
        await broadcast_train_log(line)
        log.info("SIGINT sent to training process.")
    except ProcessLookupError:
        pass

    return JSONResponse({"ok": True})


async def _run_training(cmd: list[str]):
    """Run the training subprocess and stream output to SSE subscribers."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(PROJECT_ROOT),
        )
        train_process_state["process"] = proc

        async for raw_line in proc.stdout:
            line = raw_line.decode("utf-8", errors="replace").rstrip()
            train_process_state["logs"].append(line)
            await broadcast_train_log(line)

        await proc.wait()
        train_process_state["process"] = None

        if proc.returncode == 0:
            train_process_state["status"] = "complete"
            line = "--- Training completed successfully ---"
        else:
            train_process_state["status"] = "error"
            train_process_state["error"] = f"Exit code {proc.returncode}"
            line = f"--- Training exited (code {proc.returncode}) ---"

        train_process_state["logs"].append(line)
        train_process_state["stopped_at"] = time.time()
        await broadcast_train_log(line)
        log.info(f"Training finished: {train_process_state['status']}")

    except Exception as e:
        train_process_state["status"] = "error"
        train_process_state["error"] = str(e)
        train_process_state["stopped_at"] = time.time()
        train_process_state["process"] = None
        line = f"--- Training error: {e} ---"
        train_process_state["logs"].append(line)
        await broadcast_train_log(line)
        log.error(f"Training error: {e}")


@app.get("/train/logs")
async def stream_train_logs(request: Request):
    """SSE endpoint — streams training subprocess output in real time."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=5000)
    train_log_subscribers.append(queue)

    async def event_generator():
        # Replay existing log lines first
        for line in list(train_process_state["logs"]):
            yield {"data": line}

        # Stream live
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    line = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield {"data": line}
                except asyncio.TimeoutError:
                    yield {"comment": "keepalive"}
                    if train_process_state["status"] in ("complete", "error", "idle"):
                        break
        finally:
            if queue in train_log_subscribers:
                train_log_subscribers.remove(queue)

    return EventSourceResponse(event_generator())


# ============================================================================
# Routes — Benchmark
# ============================================================================

@app.post("/bench/log")
async def receive_bench_log(request: Request):
    """Receive benchmark metrics from bench_throughput.py."""
    data = await request.json()
    event = data.get("event", "unknown")

    if event == "bench_start":
        bench_state["status"] = "running"
        bench_state["config"] = data
        bench_state["trials"] = data.get("trials", [])
        bench_state["results"] = []
        bench_state["step_log"].clear()
        bench_state["current_trial"] = None
        log.info(f"Benchmark started — {data.get('total_trials', '?')} trials")

    elif event == "trial_start":
        bench_state["current_trial"] = data.get("trial")
        log.info(f"Trial started: {data.get('trial')}")

    elif event == "bench_step":
        bench_state["step_log"].append(data)

    elif event == "trial_skip":
        bench_state["results"].append(data)
        bench_state["current_trial"] = None
        label = data.get("trial", "?")
        reason = data.get("reason", "low memory")
        log.info(f"Trial skipped: {label} — {reason}")

    elif event == "trial_end":
        bench_state["results"].append(data)
        bench_state["current_trial"] = None
        label = data.get("label", "?")
        if data.get("oom"):
            log.info(f"Trial ended: {label} — OOM")
        else:
            log.info(f"Trial ended: {label} — {data.get('tokens_per_sec', 0):,.0f} tok/s")

    elif event == "bench_end":
        bench_state["status"] = "complete"
        bench_state["results"] = data.get("results", bench_state["results"])
        log.info("Benchmark complete.")

    await broadcast_bench(data)
    return {"ok": True}


@app.get("/bench/stream")
async def stream_bench(request: Request):
    """SSE endpoint — replays step history then streams live bench events."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=5000)
    bench_subscribers.append(queue)

    async def event_generator():
        # Replay history
        for entry in list(bench_state["step_log"]):
            yield {"data": json.dumps(entry)}

        # Stream live
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield {"data": message}
                except asyncio.TimeoutError:
                    yield {"comment": "keepalive"}
        finally:
            if queue in bench_subscribers:
                bench_subscribers.remove(queue)

    return EventSourceResponse(event_generator())


@app.get("/bench/state")
async def get_bench_state():
    """Current benchmark state snapshot (for page reload)."""
    return JSONResponse({
        "status": bench_state["status"],
        "config": bench_state["config"],
        "trials": bench_state["trials"],
        "results": bench_state["results"],
        "current_trial": bench_state["current_trial"],
    })


@app.get("/bench/history")
async def get_bench_history():
    """Full per-step metrics history."""
    return JSONResponse(list(bench_state["step_log"]))


# ============================================================================
# Routes — Static Files
# ============================================================================

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/")
async def serve_dashboard():
    """Serve the dashboard HTML."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/game")
async def serve_game():
    """Serve the game adapter training dashboard."""
    return FileResponse(STATIC_DIR / "game.html")


@app.get("/bench")
async def serve_bench():
    """Serve the benchmark dashboard HTML."""
    return FileResponse(STATIC_DIR / "bench.html")


@app.get("/rl")
async def serve_rl():
    """Serve the RL training dashboard."""
    return FileResponse(STATIC_DIR / "rl.html")


@app.get("/static/{filename}")
async def serve_static(filename: str):
    """Serve static assets."""
    filepath = STATIC_DIR / filename
    if filepath.exists():
        return FileResponse(filepath)
    return JSONResponse({"error": "not found"}, status_code=404)
