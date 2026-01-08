#!/usr/bin/env bash
set -euo pipefail

SESSION="sheet_session"
WINDOW="streamer"
WORKDIR="$HOME/pod/VIXster"
CMD="python ibkr_stream_spreads.py"

# Ensure session exists
tmux has-session -t "$SESSION" 2>/dev/null || {
  echo "❌ tmux session '$SESSION' not found."
  exit 1
}

# Kill the window if it already exists (ignore error if not)
tmux kill-window -t "${SESSION}:${WINDOW}" 2>/dev/null || true

# Recreate window and run command
tmux new-window -t "$SESSION" -n "$WINDOW" -c "$WORKDIR" "$CMD"

# If we're already inside tmux, focus it
if [[ -n "${TMUX:-}" ]]; then
  tmux select-window -t "${SESSION}:${WINDOW}"
fi

echo "🚀✅ Recreated and started ${SESSION}:${WINDOW}"
