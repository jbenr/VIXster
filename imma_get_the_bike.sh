#!/bin/bash

SESSION="sheet_session"

# Kill existing session if it exists
tmux has-session -t $SESSION 2>/dev/null
if [ $? -eq 0 ]; then
  tmux kill-session -t $SESSION
fi

# Create new tmux session
tmux new-session -d -s $SESSION -n streamlit

# -------------------- Window 0: Streamlit app --------------------
tmux send-keys -t $SESSION:0 "conda activate algo" C-m
tmux send-keys -t $SESSION:0 "cd ~/pod/VIXster" C-m
tmux send-keys -t $SESSION:0 "streamlit run app.py" C-m

# -------------------- Window 1: Tailscale + Funnel --------------------
tmux new-window -t $SESSION -n tailscale
tmux send-keys -t $SESSION:1 "tailscale up --operator=\$USER || echo 'Tailscale already up'" C-m
tmux send-keys -t $SESSION:1 "sleep 3" C-m
tmux send-keys -t $SESSION:1 "tailscale funnel 8501" C-m

## -------------------- Window 2: IB Gateway (IBKR) --------------------
tmux new-window -t $SESSION -n ibkr
tmux send-keys -t $SESSION:2 "./pod/VIXster/launch_ibkr.sh" C-m
tmux send-keys -t $SESSION:2 "sleep 10" C-m

# -------------------- Window 3: Tailscale + Funnel --------------------
tmux new-window -t $SESSION -n streamer
tmux send-keys -t $SESSION:3 "conda activate sheet" C-m
tmux send-keys -t $SESSION:3 "cd ~/pod/VIXster" C-m
tmux send-keys -t $SESSION:3 "python ibkr_stream_spreads.py" C-m

# -------------------- Attach to session --------------------
tmux attach-session -t $SESSION
