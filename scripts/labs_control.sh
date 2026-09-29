#!/usr/bin/env bash
# One entry point for Labs episode restore and FastWAM inference.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

if [[ $# -eq 0 ]]; then
  set -- --help
fi
selected=false
for arg in "$@"; do
  case "$arg" in
    --restore|--infer) selected=true ;;
    -h|--help)
      cat <<'HELP'
Usage: bash scripts/labs_control.sh --restore [--infer] [options]
       bash scripts/labs_control.sh --infer [options]

  --restore    Return both arms to --start-episode's first measured pose.
               Open both grippers while returning.
  --infer      Run inference from the current pose; no implicit restore.
  Both flags   Return once, wait for arrival, then start inference.

Default: dry-run, no robot command publication.
Default endpoint: wss://u730748-b58d-17b41c61.bjb1.seetacloud.com:8443/infer.
Uses the server's default policy. No model/checkpoint/policy identity is checked.
FastWAM state/action: [left joint1..7, left gripper, right joint1..7, right gripper].
Receive, save and execute all Hx16 absolute joint targets (currently 32 rows).
At 9 Hz each chunk spans H/9 seconds plus settling; then request a fresh observation.
Use --execute-steps N to execute only the first N rows, then observe/infer again.
All predictions are still saved; unused rows are not queued. No delta or IK.
The selected format must match the service; use labs_fastwam_eef_control.sh for EEF20.
Raw state16, three 640x480 RGB images; binary fastwam.msgpack.v1 transport.
Default reference rate is 9 Hz; continuous tracking runs at 100 Hz.
For motion add BOTH --publish --enable-robot.

Options: --start-episode N (default 0), --max-chunks N (default 1),
         --execute-steps N (1..returned rows, default all),
         --dataset PATH, --url URL, --output PATH, --ik PATH,
         --speed FACTOR (default 0.3 = 9 Hz), --timeout SECONDS (default 60).

Return uses the same time scale: default duration is 1/0.3 times the base
return duration. Continuous interpolation/publication stays at 100 Hz.

Examples:
  bash scripts/labs_control.sh --restore
  bash scripts/labs_control.sh --infer
  bash scripts/labs_control.sh --restore --publish --enable-robot
  bash scripts/labs_control.sh --restore --infer --max-chunks 10 --publish --enable-robot

Dataset/server default to this Labs station; override with arguments or
LABS_DATASET / LABS_SERVER_URL. Motion requires both follower controllers
already active and exclusive command-topic ownership. This script never
switches controllers. It reuses or starts a persistent relay, which keeps
holding the last target after this client exits. Log: outputs/labs_relay/relay.log.
HELP
      exit 0 ;;
  esac
done
if [[ "$selected" != true ]]; then
  echo 'Select --restore, --infer, or both. Use --help for examples.' >&2
  exit 2
fi

# Help/argument selection works even on machines without ROS.
set +u
source "${LABS_ROS_SETUP:-/opt/ros/humble/setup.bash}"
if [[ -f site/install/setup.bash ]]; then
  source site/install/setup.bash
fi
set -u
export ROS_DOMAIN_ID="${LABS_ROS_DOMAIN_ID:-100}"
export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
export CYCLONEDDS_URI="${CYCLONEDDS_URI:-${LABS_DDS_CONFIG:-/home/agile/work/labs/deployments/example_station/cyclonedds.xml}}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
relay_options=(--manage-relay)
if [[ -f site/install/labs_fr3_kinematics/lib/labs_fr3_kinematics/labs_fr3_ik ]]; then
  relay_options+=(--ik site/install/labs_fr3_kinematics/lib/labs_fr3_kinematics/labs_fr3_ik)
fi
exec "${LABS_PYTHON:-.venv/bin/python}" -m franka_duo_tele_data.labs_client \
  --dataset "${LABS_DATASET:-/home/agile/work/labs/data/lerobot/labs_fr3_link8_delta14_20260916}" \
  --url "${LABS_SERVER_URL:-wss://u730748-b58d-17b41c61.bjb1.seetacloud.com:8443/infer}" \
  --server-profile absolute_joint16 --timeout 60 "${relay_options[@]}" "$@"
