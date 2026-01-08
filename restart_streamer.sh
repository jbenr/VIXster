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

TARGET="${SESSION}:${WINDOW}"

# If the window doesn't exist, create it (as a shell in WORKDIR)
tmux list-windows -t "$SESSION" -F '#W' | grep -qx "$WINDOW" \
  || tmux new-window -t "$SESSION" -n "$WINDOW" -c "$WORKDIR"

# Restart inside the target window/pane (pane 0)
tmux send-keys -t "${TARGET}.0" C-c
tmux send-keys -t "${TARGET}.0" "cd \"$WORKDIR\" && $CMD" C-m

# Attach/switch after the restart command is sent
if [[ -n "${TMUX:-}" ]]; then
  tmux select-window -t "$TARGET"
else
  tmux attach -t "$SESSION" \; select-window -t "$TARGET"
fi
