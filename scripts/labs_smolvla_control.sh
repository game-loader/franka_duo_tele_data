#!/usr/bin/env bash
# Labs SmolVLA: raw state20, three RGB images, absolute pose20 chunks at 9 Hz.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ $# -eq 0 ]]; then
  set -- --help
fi
for arg in "$@"; do
  case "$arg" in
    -h|--help)
      cat <<'HELP'
Usage: bash scripts/labs_smolvla_control.sh --infer [--restore] [options]
       bash scripts/labs_smolvla_control.sh --restore [options]

Endpoint: ws://100.73.14.65:8081/infer (Tailnet)
Protocol: smolvla.msgpack.v1, binary MessagePack.
Raw 20D state (dual xyz+rot6d and grippers), three 640x480 RGB PNG images.
Joint angles stay local for control/recording; no client normalization.
Execute all returned Hx20 absolute poses (currently 32 rows).
No model/checkpoint/policy identity check or /info dependency.
prediction_horizon and n_action_steps do not determine execution length.
One chunk has H/9 seconds of reference time, plus any settling tail.
Targets: [left xyz+rotation6D, right xyz+rotation6D, left gripper, right gripper].
Absolute link0/link8 poses: no delta addition or integration; rotation6D columns
are orthogonalized before IK. Gripper values are thresholded at 0.5.
9 Hz references, 100 Hz continuous tracking, target velocity weight 0.

--infer      Infer from the current pose (default one chunk).
--restore    Return once to episode start before inference, or restore only.
Dry-run by default. Motion requires BOTH --publish --enable-robot.
Options: --max-chunks N, --execute-steps N (1..returned rows), --task TEXT, --start-episode N, --output PATH,
         --dataset PATH, --url URL, --timeout SECONDS, --ik PATH.
Task defaults to the validated training-data task.

Examples:
  bash scripts/labs_smolvla_control.sh --infer
  bash scripts/labs_smolvla_control.sh --restore --infer --max-chunks 100 --publish --enable-robot

LABS_SMOLVLA_SERVER_URL overrides the endpoint. Use 127.0.0.1 only when
the model server runs on the same host as this client.
HELP
      exit 0 ;;
    --speed|--speed=*|--server-profile|--server-profile=*)
      echo 'This script fixes playback to 9 Hz and requires absolute EEF20.' >&2
      exit 2 ;;
  esac
done
exec bash scripts/labs_control.sh \
  --url "${LABS_SMOLVLA_SERVER_URL:-ws://100.73.14.65:8081/infer}" \
  --wire-protocol smolvla.msgpack.v1 "$@" --server-profile absolute20 --speed 0.3
