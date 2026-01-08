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

# Ensure window exists (create it as an interactive shell)
if ! tmux list-windows -t "$SESSION" -F '#W' | grep -qx "$WINDOW"; then
  tmux new-window -t "$SESSION" -n "$WINDOW" -c "$WORKDIR"
fi

TARGET="${SESSION}:${WINDOW}"

# Go to the right window (if you're already in tmux, just select it;
# if you're outside, attach straight into it)
if [[ -n "${TMUX:-}" ]]; then
  tmux select-window -t "$TARGET"
else
  tmux attach -t "$SESSION" \; select-window -t "$TARGET"
fi

# Now "do the things" in that window/pane:
# stop current process, cd, then start streamer
tmux send-keys -t "${TARGET}.0" C-c
tmux send-keys -t "${TARGET}.0" "cd \"$WORKDIR\"" C-m
tmux send-keys -t "${TARGET}.0" "$CMD" C-m

echo "🚀✅ Sent restart commands to $TARGET (window preserved)"
