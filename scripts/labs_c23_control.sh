#!/usr/bin/env bash
# FR3-C23 endpoint, continuous whole-chunk playback at 9 Hz. Dry-run default.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ $# -eq 0 ]]; then
  set -- --help
fi
for arg in "$@"; do
  case "$arg" in
    -h|--help)
      cat <<'HELP'
Usage: bash scripts/labs_c23_control.sh --infer [--restore] [options]
       bash scripts/labs_c23_control.sh --restore [options]

FR3-C23: 34D state, three 640x480 images, 32x14 delta actions.
Default endpoint: ws://workspace.featurize.cn:50706/infer (direct connection)
Model health must identify FastWAM-FR3-C23 / c23.
Action reference playback: 9 Hz (30 Hz model source * speed 0.3).
Each chunk accumulates deltas from its request state, row by row.
Rotations compose on the left; gripper values remain absolute.
Continuous interpolation/publication stays at 100 Hz; a chunk may have a
settling tail after its 32/9 = 3.556 s reference duration.
Return also uses speed 0.3: 3.333 times the base joint-return duration.

--infer      Infer from the current pose (default one chunk).
--restore    Return to episode start; combine with --infer to return once first.
Dry-run by default. For actual motion add BOTH --publish --enable-robot.
Options: --max-chunks N, --start-episode N, --output PATH, --dataset PATH,
         --url URL, --timeout SECONDS, --ik PATH.

Examples:
  bash scripts/labs_c23_control.sh --infer
  bash scripts/labs_c23_control.sh --infer --max-chunks 10 --publish --enable-robot
  bash scripts/labs_c23_control.sh --restore --infer --max-chunks 10 --publish --enable-robot

Set LABS_C23_SERVER_URL to override the default C23 endpoint.
Uses the same site relay and publication gates as labs_control.sh.
HELP
      exit 0 ;;
    --speed|--speed=*|--server-profile|--server-profile=*)
      echo 'This C23 script fixes playback to 9 Hz and requires the C23 model profile.' >&2
      exit 2 ;;
  esac
done
exec bash scripts/labs_control.sh \
  --url "${LABS_C23_SERVER_URL:-ws://workspace.featurize.cn:50706/infer}" \
  "$@" --server-profile c23 --speed 0.3
