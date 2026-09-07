#!/usr/bin/env bash
# Continuous SmolVLA: 9 Hz actions, infer again with 400 ms left, blend future chunks.
# Usage: bash scripts/run_smolvla_loop.sh [MAX_CHUNKS] [SPEED] [REQUEST_LEAD_MS]
# Default speed 0.3 gives 30 * 0.3 = 9 Hz on both client and servo.
# Ctrl-C stops requests; the accepted plan finishes and the servo keeps holding.
set -euo pipefail
CHUNKS="${1:-1000}"
SPEED="${2:-0.3}"
LEAD_MS="${3:-400}"
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p log/servo_start
exec 8>log/servo_start/.smolvla_loop.lock
flock -n 8 || { echo "ABORT: another SmolVLA loop is running" >&2; exit 1; }
set +u
source ~/tmr_env.sh
source site/install/setup.bash
set -u
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python3 - "$CHUNKS" "$SPEED" "$LEAD_MS" <<'PY'
import math, sys
chunks, speed, lead = int(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3])
if chunks < 1 or not math.isfinite(speed) or not 0 < speed <= 1 or not math.isfinite(lead) or lead <= 0:
    raise SystemExit("MAX_CHUNKS >= 1, SPEED in (0,1], REQUEST_LEAD_MS > 0 required")
print(f"action rate: {30 * speed:g} Hz; servo playback_speed: {speed:g}; request with {lead:g} ms left")
PY
imp() {
  local result
  result=$(timeout 10 ros2 control list_controllers -c "/$1/controller_manager" 2>/dev/null) || {
    echo "ABORT: $1 driver unavailable; run scripts/ptp_home.sh first" >&2; return 1;
  }
  printf '%s\n' "$result" | sed $'s/\033\\[[0-9;]*m//g' | awk '$1 == "joint_impedance_controller" {state=$NF} END {print state == "" ? "not_loaded" : state}'
}
servo_matches() {
  local state
  state=$(timeout 5 ros2 topic echo --once /franka_duo/joint_servo/status --field data 2>/dev/null | awk '/status_v1/ {print}') || return 1
  python3 - "$state" "$SPEED" <<'PY'
import json, math, sys
if not sys.argv[1]:
    raise SystemExit(1)
s = json.loads(sys.argv[1])
raise SystemExit(0 if (
    not s["fault"] and math.isclose(s["playback_speed"], float(sys.argv[2]), abs_tol=1e-9)
    and s.get("action_rate_hz") == 30 and s.get("commit_lead_steps") == 0
    and s.get("blend_steps") == 4 and s.get("blend_mode") == "quintic_hold_v1"
) else 1)
PY
}
left_state=$(imp left); right_state=$(imp right)
if [ "$left_state" != active ] || [ "$right_state" != active ] || ! servo_matches; then
  echo "Preparing servo/controllers with the requested playback speed and blending"
  # The detached servo/relay must not inherit the inference-loop lock.
  bash scripts/start_servo_and_activate.sh "$SPEED" --reconfigure 8>&-
fi
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
OUT="outputs/smolvla_loop_$STAMP"
mkdir -p "$OUT"
echo "logs: $OUT/  (Ctrl-C to stop requesting)"
exec .venv/bin/python -m franka_duo_tele_data.smolvla_stream \
  --dataset datasets/franka_duo_lerobot_rgb20d_v1 \
  --output "$OUT/chunks.jsonl" --max-chunks "$CHUNKS" --duration-s 0 \
  --speed "$SPEED" --request-lead-ms "$LEAD_MS" --image-format jpeg \
  --publish --enable-robot 2>&1 | tee "$OUT/console.log" | grep --line-buffered -v 'rmw_cyclonedds'
