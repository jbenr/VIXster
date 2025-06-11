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
#tmux new-window -t $SESSION -n ibkr
#
## Start Xvfb virtual display
#tmux send-keys -t $SESSION:2 "Xvfb :1 -ac -screen 0 1024x768x24 &" C-m
#tmux send-keys -t $SESSION:2 "export DISPLAY=:1" C-m
#
## Optional: Add a 2-second delay to ensure Xvfb is up
#tmux send-keys -t $SESSION:2 "sleep 2" C-m
#
## Launch IBController and IB Gateway
#tmux send-keys -t $SESSION:2 "~/ibc/gatewaystart.sh" C-m

# -------------------- Window 3: Tailscale + Funnel --------------------
tmux new-window -t $SESSION -n streamer
tmux send-keys -t $SESSION:2 "conda activate sheet" C-m
tmux send-keys -t $SESSION:2 "cd ~/pod/VIXster" C-m
tmux send-keys -t $SESSION:2 "python ibkr_stream_spreads.py" C-m

# -------------------- Attach to session --------------------
tmux attach-session -t $SESSION
