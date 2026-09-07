#!/usr/bin/env bash
# Relaunch the ZED head camera on the .50 host (tmr-user). Run as a FILE, never as an
# inline ssh command: an inline command line contains the launch string itself, so
# pgrep -f would match the ssh wrapper and kill the session.
# Usage: bash dev_zed_relaunch_50.sh
set -uo pipefail
WS="${ZED_WS:-$HOME/ros2_ws}"
LOG_DIR="$HOME/log"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/zed_$(date +%Y%m%d_%H%M%S).log"

pids() {
  ps -eo pid=,args= | awk '
    $0 ~ /zed_camera\.launch\.py/ && $0 !~ /awk/ {print $1; next}
    $0 ~ /component_container_isolated/ && $0 ~ /head_camera/ && $0 !~ /awk/ {print $1}'
}
OLD="$(pids || true)"
if [ -n "$OLD" ]; then
  echo "stopping: $(echo "$OLD" | tr '\n' ' ')"
  for P in $OLD; do kill "$P" 2>/dev/null || true; done
  sleep 3
  for P in $(pids || true); do kill -9 "$P" 2>/dev/null || true; done
  sleep 1
fi
echo "remaining: $(pids | wc -l)"

set +u
source /opt/ros/jazzy/setup.bash >/dev/null 2>&1
source "$WS/install/setup.bash" >/dev/null 2>&1
set -u
nohup setsid ros2 launch zed_wrapper zed_camera.launch.py \
  camera_model:=zedm namespace:=head_camera publish_tf:=false serial_number:=17064700 \
  > "$LOG" 2>&1 < /dev/null &
echo "launched -> $LOG"
sleep 30
echo "procs: $(pids | wc -l)"
grep -iE 'Camera Model|successfully opened|error|failed|died' "$LOG" | tail -3 | cut -c1-150
printf 'hz: '
timeout 10 ros2 topic hz /head_camera/zed/rgb/color/rect/image 2>&1 | grep -m1 -E 'average|no new' || echo NONE
