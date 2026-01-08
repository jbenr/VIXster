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

# Ensure the window exists; if not, create it (but do NOT kill/delete it)
if ! tmux list-windows -t "$SESSION" -F '#W' | grep -qx "$WINDOW"; then
  tmux new-window -t "$SESSION" -n "$WINDOW" -c "$WORKDIR"
fi

# Restart the process *inside* the existing window (pane 0) without deleting the window
# NOTE: If your streamer is in a different pane, change .0 to .1, .2, etc.
tmux respawn-pane -t "${SESSION}:${WINDOW}.0" -k "bash -lc 'cd \"$WORKDIR\" && $CMD; echo; echo \"[streamer exited — press enter]\"; read'"

# If we're already inside tmux, focus it
if [[ -n "${TMUX:-}" ]]; then
  tmux select-window -t "${SESSION}:${WINDOW}"
fi

echo "🚀✅ Restarted ${SESSION}:${WINDOW} (window preserved)"
