#!/usr/bin/env bash
# Absolute joint16 remote policy, variable chunk length at 9 Hz. Dry-run by default.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ $# -eq 0 ]]; then set -- --help; fi
infer=false
has_url=false
for arg in "$@"; do
  case "$arg" in
    -h|--help)
      cat <<'HELP'
Usage: bash scripts/labs_joint16_control.sh --infer --url ws://HOST:PORT/infer [options]
       bash scripts/labs_joint16_control.sh --restore [options]

State/action: [left joint1..7, left gripper, right joint1..7, right gripper].
Raw absolute joint angles in radians; grippers 0=closed, 1=open.
Three RGB 640x480 images: head, wrist_left, wrist_right; optional --task TEXT.
Binary MessagePack; default subprotocol smolvla.msgpack.v1, override with
--wire-protocol NAME (--joint16-protocol NAME remains an alias).
No model/checkpoint/policy identity check or /info dependency.
Execute all returned Hx16 absolute targets, with no delta accumulation or IK.
Use --execute-steps N to execute only the first N returned rows.
9 Hz references, 100 Hz continuous joint tracking; no client normalization.

--restore returns once to the measured start of --start-episode (default 0).
--infer requests model chunks (default 1); --max-chunks N changes the count.
Motion requires BOTH --publish --enable-robot. Otherwise dry-run.
All replies/targets/feedback are saved in outputs/.../actions.msgpack.

LABS_JOINT16_SERVER_URL sets the endpoint; no Cartesian-model URL is assumed.
LABS_JOINT16_DATASET overrides the joint16 restore/contract dataset.

Example:
  bash scripts/labs_joint16_control.sh --restore --infer --url ws://HOST:PORT/infer \
    --max-chunks 100 --publish --enable-robot
HELP
      exit 0 ;;
    --infer) infer=true ;;
    --url|--url=*) has_url=true ;;
    --server-profile|--server-profile=*|--speed|--speed=*)
      echo 'This entry point requires joint16 and fixes reference playback to 9 Hz.' >&2
      exit 2 ;;
  esac
done
url_options=()
if [[ -n "${LABS_JOINT16_SERVER_URL:-}" ]]; then
  url_options=(--url "$LABS_JOINT16_SERVER_URL")
  has_url=true
fi
if [[ "$infer" == true && "$has_url" != true ]]; then
  echo 'Set --url ws://HOST:PORT/infer or LABS_JOINT16_SERVER_URL for the joint16 model.' >&2
  exit 2
fi
forwarded=()
for arg in "$@"; do
  case "$arg" in
    --joint16-protocol) forwarded+=(--wire-protocol) ;;
    --joint16-protocol=*) forwarded+=("--wire-protocol=${arg#*=}") ;;
    *) forwarded+=("$arg") ;;
  esac
done
exec bash scripts/labs_control.sh \
  --dataset "${LABS_JOINT16_DATASET:-/home/agile/work/labs/data/lerobot_joint16/labs_fr3_joint16_20260916}" \
  --wire-protocol smolvla.msgpack.v1 "${url_options[@]}" "${forwarded[@]}" \
  --server-profile absolute_joint16 --speed 0.3
