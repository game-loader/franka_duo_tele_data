#!/usr/bin/env bash
# Return through the already-running site servo; no PTP or MCAP.
set -euo pipefail
PUBLISH=0
ENABLE_ROBOT=0
for arg in "$@"; do
  case "$arg" in
    --publish) PUBLISH=1 ;;
    --enable-robot) ENABLE_ROBOT=1 ;;
    --help|-h)
      cat <<'HELP'
Usage: bash scripts/return_to_start.sh [--publish --enable-robot]
Default: read live poses/grippers and save a motion plan without publishing.
Both flags: move through the current servo to episode 0 frame 0 observation.state.
Preserves current physical gripper opening. No PTP, inference, cameras or MCAP.
Stop the policy loop first and wait for the servo to hold.
The servo must already be running (scripts/start_servo_and_activate.sh 0.3).

Environment overrides:
  DELTA14_DATASET  Original RGB20D dataset metadata root
  DELTA14_SPEED    Must match servo playback speed (default 0.3 = 9 Hz)
  TMR_ENV_FILE    Host-managed ROS environment (default ~/tmr_env.sh)

Ctrl-C stops new chunks; already accepted targets can finish before servo hold.
HELP
      exit 0 ;;
    *) echo "Unknown option: $arg (see --help)" >&2; exit 2 ;;
  esac
done
[[ "$PUBLISH" = "$ENABLE_ROBOT" ]] || {
  echo "Robot execution requires both --publish and --enable-robot" >&2; exit 2;
}
cd "$(dirname "${BASH_SOURCE[0]}")/.."
[[ -x .venv/bin/python ]] || { echo "Missing .venv/bin/python; install inference-client dependencies" >&2; exit 1; }
set +u
source "${TMR_ENV_FILE:-$HOME/tmr_env.sh}"
source site/install/setup.bash
set -u
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
# The Python entrypoint takes the shared inference lock, including direct invocations.
exec .venv/bin/python -m franka_duo_tele_data.return_to_start \
  --dataset "${DELTA14_DATASET:-datasets/franka_duo_lerobot_rgb20d_v1}" \
  --speed "${DELTA14_SPEED:-0.3}" "$@"
