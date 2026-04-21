#!/bin/bash
set -euo pipefail

PROJECT_DIR="$HOME/Desktop/otonom"
WAIT_SECONDS=5
LOG_FILE="/tmp/gz_sim_otonom.log"

MAIN_PID=$$
rm -f /tmp/gz_sim_otonom_pids

# 1. Source user profiles and ArduPilot completion
[ -f "$HOME/ardupilot/Tools/completion/completion.bash" ] && source "$HOME/ardupilot/Tools/completion/completion.bash"
[ -f "$HOME/.profile" ] && source "$HOME/.profile"
[ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc"

# 2. Export all safety-net Gazebo environment variables
export GZ_VERSION="${GZ_VERSION:-harmonic}"
export GZ_SIM_SYSTEM_PLUGIN_PATH="/usr/local/lib/ardupilot_gazebo:${GZ_SIM_SYSTEM_PLUGIN_PATH:-}"
export GZ_SIM_RESOURCE_PATH="$HOME/SITL_Models/Gazebo/models:$HOME/SITL_Models/Gazebo/worlds:${GZ_SIM_RESOURCE_PATH:-}"
export IGN_GAZEBO_RESOURCE_PATH="$HOME/SITL_Models/Gazebo/models:$HOME/SITL_Models/Gazebo/worlds:${IGN_GAZEBO_RESOURCE_PATH:-}"
export GAZEBO_MODEL_PATH="$HOME/SITL_Models/Gazebo/models:${GAZEBO_MODEL_PATH:-}"

mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

echo "--> Starting Gazebo..."
# 3. Start Gazebo detached
setsid bash -lc '
[ -f "$HOME/.profile" ] && source "$HOME/.profile"
[ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc"
export GZ_VERSION="${GZ_VERSION:-harmonic}"
export GZ_SIM_SYSTEM_PLUGIN_PATH="/usr/local/lib/ardupilot_gazebo:${GZ_SIM_SYSTEM_PLUGIN_PATH:-}"
export GZ_SIM_RESOURCE_PATH="$HOME/SITL_Models/Gazebo/models:$HOME/SITL_Models/Gazebo/worlds:${GZ_SIM_RESOURCE_PATH:-}"
export IGN_GAZEBO_RESOURCE_PATH="$HOME/SITL_Models/Gazebo/models:$HOME/SITL_Models/Gazebo/worlds:${IGN_GAZEBO_RESOURCE_PATH:-}"
export GAZEBO_MODEL_PATH="$HOME/SITL_Models/Gazebo/models:${GAZEBO_MODEL_PATH:-}"
exec gz sim -v4 -r "$HOME/SITL_Models/Gazebo/worlds/world_two_talons.sdf"
' >"$LOG_FILE" 2>&1 &

GZ_PID=$!

cleanup() {
    echo "Shutting down everything..."
    kill -- -"$GZ_PID" 2>/dev/null || kill "$GZ_PID" 2>/dev/null || true
    if [ -f /tmp/gz_sim_otonom_pids ]; then
        while read -r pid; do
            kill -TERM "$pid" 2>/dev/null || true
        done < /tmp/gz_sim_otonom_pids
        rm -f /tmp/gz_sim_otonom_pids
    fi
    killall -9 sim_vehicle.py mavproxy.py ArduPlane 2>/dev/null || true
}
trap cleanup EXIT INT TERM

sleep "$WAIT_SECONDS"

# 4. Define Commands for Plane 1 and Plane 2
PLANE1_CMD='
echo $$ >> /tmp/gz_sim_otonom_pids
trap "trap - EXIT; kill -TERM '"$MAIN_PID"' 2>/dev/null || true; exit" EXIT INT TERM HUP
[ -f "$HOME/.profile" ] && source "$HOME/.profile"
[ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc"
cd ~/ardupilot
echo "Starting Plane 1 (Observer)..."
sim_vehicle.py -v ArduPlane -f JSON --model JSON:127.0.0.1:9012 --add-param-file=$HOME/SITL_Models/Gazebo/config/mini_talon_vtail.param --map -I0 --out=127.0.0.1:14580 --custom-location=38.700907,27.453879,10,90 --sysid 1
'

PLANE2_CMD='
echo $$ >> /tmp/gz_sim_otonom_pids
trap "trap - EXIT; kill -TERM '"$MAIN_PID"' 2>/dev/null || true; exit" EXIT INT TERM HUP
[ -f "$HOME/.profile" ] && source "$HOME/.profile"
[ -f "$HOME/.bashrc" ] && source "$HOME/.bashrc"
cd ~/ardupilot
echo "Starting Plane 2 (Target)..."
sim_vehicle.py -v ArduPlane -f JSON --model JSON:127.0.0.1:9012 --add-param-file=$HOME/SITL_Models/Gazebo/config/mini_talon_vtail.param --map -I1 --out=127.0.0.1:14581 --custom-location=38.700907,27.454020,0,90 --sysid 2
'

echo "--> Spawning ArduPilot instances in new terminals..."

# 5. Launch Terminals based on system availability
if command -v gnome-terminal >/dev/null 2>&1; then
    gnome-terminal --tab -- bash -lc "$PLANE1_CMD"
    sleep 2
    gnome-terminal --tab -- bash -lc "$PLANE2_CMD"
elif command -v konsole >/dev/null 2>&1; then
    konsole --new-tab -e bash -lc "$PLANE1_CMD" &
    sleep 2
    konsole --new-tab -e bash -lc "$PLANE2_CMD" &
elif command -v xfce4-terminal >/dev/null 2>&1; then
    xfce4-terminal --tab -e "bash -lc '$PLANE1_CMD'" &
    sleep 2
    xfce4-terminal --tab -e "bash -lc '$PLANE2_CMD'" &
elif command -v xterm >/dev/null 2>&1; then
    xterm -hold -e "bash -lc '$PLANE1_CMD'" &
    sleep 2
    xterm -hold -e "bash -lc '$PLANE2_CMD'" &
else
    echo "No supported GUI terminal found. Running in current window (might cause layout issues)."
    bash -lc "$PLANE1_CMD" &
    sleep 2
    bash -lc "$PLANE2_CMD" &
fi

echo ""
echo "========================================="
echo "Gazebo is running in the background."
echo "Plane 1 and Plane 2 have been launched in separate terminal windows."
echo "Leave this window open. Press [CTRL+C] here to kill Gazebo and exit."
echo "========================================="

# Keep the script running to hold the Gazebo process alive
wait