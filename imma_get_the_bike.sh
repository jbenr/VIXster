#!/bin/bash

SESSION="sheet_session"

# Kill existing session if it exists
tmux has-session -t $SESSION 2>/dev/null
if [ $? -eq 0 ]; then
  tmux kill-session -t $SESSION
fi

# Create new tmux session
tmux new-session -d -s $SESSION -n site

# -------------------- Window 0: VIXster site (Dash, vixster/ package) --------------------
# Old Streamlit app is still in app.py if ever needed: conda activate algo && streamlit run app.py
tmux send-keys -t $SESSION:0 "cd ~/pod/VIXster && PYTHONPATH=\$PWD ~/miniconda3/envs/vixster2/bin/python -m vixster.cli serve --port 8501" C-m

# -------------------- Window 1: Tailscale + Funnel --------------------
tmux new-window -t $SESSION -n tailscale
tmux send-keys -t $SESSION:1 "tailscale up --operator=\$USER || echo 'Tailscale already up'" C-m
tmux send-keys -t $SESSION:1 "sleep 3" C-m
tmux send-keys -t $SESSION:1 "tailscale funnel 8501" C-m

## -------------------- Window 2: IB Gateway (IBKR) --------------------
tmux new-window -t $SESSION -n ibkr
tmux send-keys -t $SESSION:0 "conda activate algo" C-m
tmux send-keys -t $SESSION:2 "./pod/VIXster/launch_ibkr.sh" C-m
tmux send-keys -t $SESSION:2 "sleep 10" C-m

# -------------------- Window 3: Tailscale + Funnel --------------------
tmux new-window -t $SESSION -n streamer
tmux send-keys -t $SESSION:3 "cd ~/pod/VIXster && PYTHONPATH=\$PWD ~/miniconda3/envs/vixster2/bin/python -m vixster.cli stream" C-m

# -------------------- Attach to session --------------------
tmux attach-session -t $SESSION
