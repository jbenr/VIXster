#!/usr/bin/env bash
# ~/pod/VIXster/restart_streamer.sh

SESSION="sheet_session"
WINDOW="streamer"            # tmux window name
SCRIPT="python ibkr_stream_spreads.py"
WORKDIR="$HOME/pod/VIXster"

tmux attach -t "${SESSION}" \; select-window -t $SESSION:2
tmux send-keys -t $SESSION:2 C-c
tmux send-keys -t $SESSION:2 C-c
sleep 1
tmux send-keys -t $SESSION:2 "python ibkr_stream_spreads.py" C-m

echo "🚀✅ Restarted streamer."
