#!/usr/bin/env bash
# ============================================================================
# Dashboard start/stop script
# Usage:
#   ./dashboard.sh start   — Start the dashboard server (background)
#   ./dashboard.sh stop    — Stop the dashboard server
#   ./dashboard.sh status  — Check if running
#   ./dashboard.sh restart — Stop then start
# ============================================================================

set -euo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
PIDFILE="$DIR/.dashboard.pid"
LOGFILE="$DIR/dashboard.log"
HOST="0.0.0.0"
PORT="28000"
VENV="$DIR/.venv"

start_dashboard() {
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        echo "Dashboard already running (PID $(cat "$PIDFILE"))"
        echo "  → http://localhost:$PORT"
        return 0
    fi

    # Activate venv if present
    if [ -d "$VENV" ]; then
        source "$VENV/bin/activate"
    fi

    echo "Starting dashboard on http://$HOST:$PORT ..."
    cd "$DIR"
    nohup python -m uvicorn dashboard.server:app \
        --host "$HOST" --port "$PORT" \
        > "$LOGFILE" 2>&1 &

    echo $! > "$PIDFILE"
    sleep 1

    if kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        echo "Dashboard started (PID $(cat "$PIDFILE"))"
        echo "  → http://localhost:$PORT"
        echo "  → Logs: $LOGFILE"
    else
        echo "Failed to start. Check $LOGFILE"
        rm -f "$PIDFILE"
        return 1
    fi
}

stop_dashboard() {
    if [ ! -f "$PIDFILE" ]; then
        echo "No PID file found — dashboard not running"
        return 0
    fi

    PID=$(cat "$PIDFILE")
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping dashboard (PID $PID)..."
        kill "$PID"
        # Wait up to 5 seconds for clean exit
        for i in $(seq 1 10); do
            if ! kill -0 "$PID" 2>/dev/null; then
                break
            fi
            sleep 0.5
        done
        # Force kill if still alive
        if kill -0 "$PID" 2>/dev/null; then
            kill -9 "$PID" 2>/dev/null || true
        fi
        echo "Dashboard stopped."
    else
        echo "Dashboard not running (stale PID file)."
    fi
    rm -f "$PIDFILE"
}

status_dashboard() {
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        echo "Dashboard is running (PID $(cat "$PIDFILE"))"
        echo "  → http://localhost:$PORT"
    else
        echo "Dashboard is not running"
        rm -f "$PIDFILE" 2>/dev/null || true
    fi
}

case "${1:-}" in
    start)   start_dashboard ;;
    stop)    stop_dashboard ;;
    restart) stop_dashboard; start_dashboard ;;
    status)  status_dashboard ;;
    *)
        echo "Usage: $0 {start|stop|status|restart}"
        exit 1
        ;;
esac
