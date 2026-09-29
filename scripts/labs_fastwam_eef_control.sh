#!/usr/bin/env bash
# FastWAM absolute link8 pose20 at 9 Hz, independent of checkpoint/model identity.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# Edit this address when moving the 20D model to another server.
# The server selects its default model; no model name or health URL is needed.
INFERENCE_URL="${LABS_FASTWAM_EEF_SERVER_URL:-wss://u730748-b58d-17b41c61.bjb1.seetacloud.com:8443/infer}"

if [[ $# -eq 0 ]]; then set -- --help; fi
for arg in "$@"; do
  case "$arg" in
    -h|--help)
      cat <<'HELP'
Usage: bash scripts/labs_fastwam_eef_control.sh --infer --task 1|2|3|4 [--restore] [options]
       bash scripts/labs_fastwam_eef_control.sh --restore --task 1|2|3|4 [options]

Binary fastwam.msgpack.v1. Uses the server's default policy.
Endpoint: edit INFERENCE_URL near the top of this script, or pass --url URL.
No model/checkpoint/policy identity check or /health dependency.
Request: state_format=rot6d_cols20, action_format=absolute20; no policy name.
Raw state20 + head/wrist_left/wrist_right RGB 640x480 images + integer task_id.
Output: all returned Hx20 absolute poses; no delta addition or inverse normalization.
Order: [left xyz+rotation6D, right xyz+rotation6D, left gripper, right gripper].
Poses are each arm's link0-to-link8; rotation6D is the first two matrix columns.
The existing IK and site follower relay continuously track the complete chunk.
9 Hz reference (H/9 seconds), 100 Hz tracking, grippers thresholded at 0.5.
--restore returns to the selected task's episode 0/frame 0 measured arm joints.
Task datasets are configured in configs/labs_fr3_31/task_starts.json.
Grippers keep their current opening. --infer starts at the current pose.
Dry-run by default; motion requires BOTH --publish --enable-robot.
Tasks (sent as integer task_id; the server selects instruction/text embeddings):
  1  Use the left arm to place the square head into the yellow box on the left, and the right arm to place the screw into the green box on the right.
  2  Open the drawer, pick up the white charger and place it inside the drawer, then close the drawer.
  3  Stack the three bowls together.
  4  Fold the towel.
--infer and --restore require --task 1, 2, 3 or 4; there is no implicit task selection.
--task-id is an alias for --task. No task text is sent.
Options: --max-chunks N, --execute-steps N (1..returned rows), --dataset PATH,
         --url URL, --output PATH, --timeout SECONDS (default 60), --ik PATH.
LABS_FASTWAM_EEF_SERVER_URL overrides the script default; --url overrides both.

Examples:
  bash scripts/labs_fastwam_eef_control.sh --infer --task 1
  bash scripts/labs_fastwam_eef_control.sh --restore --task 3 --publish --enable-robot
  bash scripts/labs_fastwam_eef_control.sh --infer --task 3 --max-chunks 100 --publish --enable-robot
HELP
      exit 0 ;;
    --server-profile|--server-profile=*|--speed|--speed=*)
      echo 'This script selects FastWAM EEF20 and fixes reference playback to 9 Hz.' >&2
      exit 2 ;;
  esac
done
selected=false
task_id=""
forwarded_args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --task|--task-id)
      if [[ $# -lt 2 ]]; then
        echo '--task requires 1, 2, 3 or 4.' >&2
        exit 2
      fi
      task_id="$2"
      shift 2 ;;
    --task=*|--task-id=*) task_id="${1#*=}"; shift ;;
    --infer|--restore) selected=true; forwarded_args+=("$1"); shift ;;
    *) forwarded_args+=("$1"); shift ;;
  esac
done
case "$task_id" in
  1|2|3|4) ;;
  "")
    if [[ "$selected" == true ]]; then
      echo 'Select a task with --task 1, 2, 3 or 4. Use --help for the instructions.' >&2
      exit 2
    fi ;;
  *) echo '--task must be 1, 2, 3 or 4. Use --help for the instructions.' >&2; exit 2 ;;
esac
if [[ -n "$task_id" ]]; then
  forwarded_args+=(--task-id "$task_id")
fi
exec bash scripts/labs_control.sh \
  --url "$INFERENCE_URL" \
  --wire-protocol fastwam.msgpack.v1 --timeout 60 "${forwarded_args[@]}" --server-profile absolute20 --speed 0.3
