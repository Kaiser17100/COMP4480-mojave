#!/bin/bash

set -euo pipefail

PROJECT_DIR="$HOME/Desktop/otonom"
WAIT_SECONDS=3
LOG_FILE="/tmp/gz_sim_otonom.log"

[ -f "$HOME/.profile" ] && source "$HOME/.profile"
[ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc"

export GZ_VERSION="${GZ_VERSION:-harmonic}"
export GZ_SIM_SYSTEM_PLUGIN_PATH="/usr/local/lib/ardupilot_gazebo:${GZ_SIM_SYSTEM_PLUGIN_PATH:-}"
export GZ_SIM_RESOURCE_PATH="$HOME/SITL_Models/Gazebo/models:$HOME/SITL_Models/Gazebo/worlds:${GZ_SIM_RESOURCE_PATH:-}"
export IGN_GAZEBO_RESOURCE_PATH="$HOME/SITL_Models/Gazebo/models:$HOME/SITL_Models/Gazebo/worlds:${IGN_GAZEBO_RESOURCE_PATH:-}"
export GAZEBO_MODEL_PATH="$HOME/SITL_Models/Gazebo/models:${GAZEBO_MODEL_PATH:-}"

mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

setsid bash -lc '
[ -f "$HOME/.profile" ] && source "$HOME/.profile"
[ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc"
export GZ_VERSION="${GZ_VERSION:-harmonic}"
export GZ_SIM_SYSTEM_PLUGIN_PATH="/usr/local/lib/ardupilot_gazebo:${GZ_SIM_SYSTEM_PLUGIN_PATH:-}"
export GZ_SIM_RESOURCE_PATH="$HOME/SITL_Models/Gazebo/models:$HOME/SITL_Models/Gazebo/worlds:${GZ_SIM_RESOURCE_PATH:-}"
export IGN_GAZEBO_RESOURCE_PATH="$HOME/SITL_Models/Gazebo/models:$HOME/SITL_Models/Gazebo/worlds:${IGN_GAZEBO_RESOURCE_PATH:-}"
export GAZEBO_MODEL_PATH="$HOME/SITL_Models/Gazebo/models:${GAZEBO_MODEL_PATH:-}"
exec gz sim -v4 -r "$HOME/SITL_Models/Gazebo/worlds/world.sdf"
' >"$LOG_FILE" 2>&1 &

GZ_PID=$!

sleep "$WAIT_SECONDS"

ARDUPILOT_CMD='
[ -f "$HOME/.profile" ] && source "$HOME/.profile"
[ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc"

cleanup() {
    echo "Gazebo kapatiliyor..."
    kill -- -'"$GZ_PID"' 2>/dev/null || kill '"$GZ_PID"' 2>/dev/null || true
}

trap cleanup EXIT INT TERM

cd ~/ardupilot
sim_vehicle.py -v ArduPlane -f JSON --model JSON --add-param-file=$HOME/SITL_Models/Gazebo/config/mini_talon_vtail.param --console --map -I0 --out=udp:127.0.0.1:14580 --custom-location=38.700907,27.453879,10,90
'

if command -v gnome-terminal >/dev/null 2>&1; then
    gnome-terminal --tab -- bash -lc "$ARDUPILOT_CMD"
elif command -v konsole >/dev/null 2>&1; then
    konsole -e bash -lc "$ARDUPILOT_CMD" &
elif command -v xfce4-terminal >/dev/null 2>&1; then
    xfce4-terminal --hold -e "bash -lc '$ARDUPILOT_CMD'" &
elif command -v xterm >/dev/null 2>&1; then
    xterm -hold -e "bash -lc '$ARDUPILOT_CMD'" &
else
    bash -lc "$ARDUPILOT_CMD"
fi
