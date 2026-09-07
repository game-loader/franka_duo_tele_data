#!/usr/bin/env bash
# One-shot cup/bowl grasp through the joint servo + impedance controllers.
# Prepares servo/relay/impedance (via start_servo_and_activate.sh) when needed, then
# runs franka_duo_tele_data.grasp_cup_bowl. Dry-run unless EXECUTE=1.
#
# Usage: [EXECUTE=1] bash scripts/run_grasp_cup_bowl.sh [cup|bowl] [right|left] [SPEED] [extra args...]
#   e.g.  bash scripts/run_grasp_cup_bowl.sh cup right 0.1            # dry-run, live camera
#         EXECUTE=1 bash scripts/run_grasp_cup_bowl.sh cup right 0.1  # moves the robot
set -euo pipefail
TARGET="${1:-cup}"; ARM="${2:-right}"; SPEED="${3:-0.1}"; shift $(( $# > 3 ? 3 : $# )) || true
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p outputs log/servo_start
set +u
source "${TMR_ENV_FILE:-$HOME/tmr_env.sh}" >/dev/null 2>&1
source site/install/setup.bash
set -u
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
imp() { timeout 10 ros2 control list_controllers -c "/$1/controller_manager" 2>/dev/null | sed $'s/\033\[[0-9;]*m//g' | awk '$1 == "joint_impedance_controller" {s=$NF} END {print s == "" ? "not_loaded" : s}'; }
servo_matches() {
  local st; st=$(timeout 5 ros2 topic echo --once /franka_duo/joint_servo/status --field data 2>/dev/null | awk '/status_v1/') || return 1
  python3 - "$st" "$SPEED" <<'PY'
import json, math, sys
if not sys.argv[1]: raise SystemExit(1)
s = json.loads(sys.argv[1])
raise SystemExit(0 if (not s["fault"] and math.isclose(s["playback_speed"], float(sys.argv[2]), abs_tol=1e-9)
    and s.get("commit_lead_steps") == 0 and s.get("blend_steps") == 4) else 1)
PY
}
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="outputs/grasp_${TARGET}_${ARM}_$STAMP.json"
if [ "${EXECUTE:-0}" = "1" ]; then
  if [ "$(imp left)" != active ] || [ "$(imp right)" != active ] || ! servo_matches; then
    echo "Preparing servo/relay/impedance at speed $SPEED"
    bash scripts/start_servo_and_activate.sh "$SPEED" --reconfigure
  fi
  echo "EXECUTING grasp: $TARGET with $ARM arm at speed $SPEED -> $OUT"
  exec .venv/bin/python -m franka_duo_tele_data.grasp_cup_bowl --dataset datasets/franka_duo_lerobot_rgb20d_v1 \
    --target "$TARGET" --arm "$ARM" --speed "$SPEED" --output "$OUT" --publish --enable-robot "$@" 2>&1 | grep --line-buffered -v rmw_cyclonedds
fi
echo "DRY-RUN (set EXECUTE=1 to move): $TARGET with $ARM arm -> $OUT"
exec .venv/bin/python -m franka_duo_tele_data.grasp_cup_bowl --dataset datasets/franka_duo_lerobot_rgb20d_v1 \
  --target "$TARGET" --arm "$ARM" --speed "$SPEED" --output "$OUT" "$@" 2>&1 | grep --line-buffered -v rmw_cyclonedds
