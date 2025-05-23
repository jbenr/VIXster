#!/usr/bin/env bash
# ~/bin/launch_ibkr.sh

set -xe  # echo each command, exit on errors

# ─── Load ~/.env into the environment ───
ENV_FILE="$HOME/pod/VIXster/.env"
if [[ -f "$ENV_FILE" ]]; then
  set -o allexport
  source "$ENV_FILE"
  set +o allexport
fi

# ─── 2. Validate presence of credentials ───
: "${IB_USER:?❌ IB_USER is not set. Please add IB_USER=… to your .env}"
: "${IB_PASS:?❌ IB_PASS is not set. Please add IB_PASS=… to your .env}"

# ─── Clean up any stale Xvfb lock on :1 ───
LOCK="/tmp/.X1-lock"
if [[ -e "$LOCK" ]]; then
  echo "🗑 Removing stale lock $LOCK"
  rm -f "$LOCK"
fi

# ─── Start Xvfb with GLX support on display :1 ───
Xvfb :1 -ac -screen 0 1024x768x24 &
XVFB_PID=$!

export DISPLAY=:1
export LIBGL_ALWAYS_SOFTWARE=1
export JAVA_TOOL_OPTIONS="-Dprism.order=sw"

sleep 2

# ─── Launch IB Gateway ───
~/Jts/ibgateway/1030/ibgateway &
IBGW_PID=$!

# ─── Wait for any window with “Gateway” in its title ───
for i in {1..15}; do
  WIDS=$(xdotool search --onlyvisible --name "Gateway")
  if [[ -n "$WIDS" ]]; then
    WID=$(echo "$WIDS" | head -n1)
    echo "✅ Found IB Gateway window: $WID"
    break
  fi
  sleep 1
done

if [[ -z "$WID" ]]; then
  echo "❌ Couldn’t find the IB Gateway window. Exiting."
  kill $IBGW_PID $XVFB_PID
  exit 1
fi

# ─── Type credentials directly into that window ───
xdotool type  --delay 5 --window "$WID" "$IB_USER"
xdotool key   --window "$WID" Tab
xdotool type  --delay 5 --window "$WID" "$IB_PASS"
xdotool key   --window "$WID" Return

# ─── Wait for Gateway to exit, then clean up Xvfb ───
wait $IBGW_PID
kill $XVFB_PID
