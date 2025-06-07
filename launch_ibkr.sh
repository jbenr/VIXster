#!/usr/bin/env bash
#set -xe

# Load env
ENV_FILE="$HOME/pod/VIXster/.env"
if [[ -f "$ENV_FILE" ]]; then
  set -o allexport
  source "$ENV_FILE"
  set +o allexport
fi

: "${IB_USER:?Missing IB_USER}"
: "${IB_PASS:?Missing IB_PASS}"

IB_FOLDER="$HOME/Jts/ibgateway/1030"
LOGFILE="$IB_FOLDER/ibgw.log"

# Graphics/env fixes
unset JAVA_TOOL_OPTIONS
export LIBGL_ALWAYS_SOFTWARE=1
export GALLIUM_DRIVER=softpipe
export MESA_GL_VERSION_OVERRIDE=3.0
export MESA_GLSL_VERSION_OVERRIDE=130
export JAVA_TOOL_OPTIONS="-Dprism.verbose=true -Dprism.order=sw -Dprism.text=t2k"

# Start Xvfb FIRST
#pkill -f "Xvfb :99" || true
#rm -f /tmp/.X99-lock

Xvfb :99 -screen 0 1024x768x24 -nolisten tcp &
XVFB_PID=$!
export DISPLAY=:99
sleep 2

# Start IB Gateway
"$IB_FOLDER/ibgateway" > "$LOGFILE" 2>&1 &
IBGW_PID=$!
echo "🚀 Launched IB Gateway (PID $IBGW_PID), DISPLAY $DISPLAY"

# Wait for GUI to be up
sleep 5

# Inject keystrokes
xdotool type "$IB_USER"
xdotool key Tab
xdotool type "$IB_PASS"
xdotool key Return

echo "✅ Sent login keystrokes to IB Gateway window."
