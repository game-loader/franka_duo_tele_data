#!/usr/bin/env bash
# FastWAM delta14 streaming through the already-started site joint servo.
# Default: one dry-run observation, no robot commands or MCAP recording.
# Execute: bash scripts/run_delta14_loop.sh --publish --enable-robot
# Each 32-step chunk finishes before the next observation/inference request.
set -euo pipefail
PUBLISH=0
ENABLE_ROBOT=0
for arg in "$@"; do
  case "$arg" in
    --publish) PUBLISH=1 ;;
    --enable-robot) ENABLE_ROBOT=1 ;;
    --help|-h)
      cat <<'HELP'
Usage: bash scripts/run_delta14_loop.sh [--publish --enable-robot]
Both flags are required for robot motion. Without them, capture one observation,
infer and convert targets, and exit without command publication. No MCAP recording.
Execution waits for all 32 steps and servo hold before requesting the next chunk.
Start servo first: bash scripts/start_servo_and_activate.sh 0.3

Environment overrides:
  FASTWAM_URL       WSS /infer endpoint
  DELTA14_DATASET   Original 30 Hz RGB20D dataset root
  DELTA14_SPEED     Playback speed, must match servo (default 0.3 = 9 Hz)
  DELTA14_CHUNKS    Maximum chunks when executing (default 1000)
  DELTA14_TIMEOUT   Inference timeout (default 10 seconds)
  TMR_ENV_FILE     Host-managed ROS environment (default ~/tmr_env.sh)

Ctrl-C stops new requests. An accepted plan can finish before the servo holds.
HELP
      exit 0 ;;
    *) echo "Unknown option: $arg (see --help)" >&2; exit 2 ;;
  esac
done
[[ "$PUBLISH" = "$ENABLE_ROBOT" ]] || {
  echo "Robot execution requires both --publish and --enable-robot" >&2; exit 2;
}
cd "$(dirname "${BASH_SOURCE[0]}")/.."
URL="${FASTWAM_URL:-wss://u730748-7892a859b4e0.bjb2.seetacloud.com:8443/infer}"
DATASET="${DELTA14_DATASET:-datasets/franka_duo_lerobot_rgb20d_v1}"
SPEED="${DELTA14_SPEED:-0.3}"
CHUNKS="${DELTA14_CHUNKS:-1000}"
INFER_TIMEOUT="${DELTA14_TIMEOUT:-10}"
[[ -x .venv/bin/python ]] || { echo "Missing .venv/bin/python; install inference-client dependencies" >&2; exit 1; }
[[ -f "$DATASET/meta/info.json" ]] || { echo "Dataset metadata missing: $DATASET; set DELTA14_DATASET" >&2; exit 1; }
set +u
source "${TMR_ENV_FILE:-$HOME/tmr_env.sh}"
source site/install/setup.bash
set -u
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p log/servo_start
# Share the existing policy wrapper lock; detached servos do not inherit it.
exec 8>log/servo_start/.smolvla_loop.lock
flock -n 8 || { echo "Another policy loop is running" >&2; exit 1; }
FLAGS=()
if [[ "$PUBLISH" = 1 ]]; then
  FLAGS=(--publish --enable-robot)
fi
STAMP="$(date +%Y%m%d_%H%M%S)_$$"
OUT="outputs/delta14_loop_$STAMP"
mkdir -p "$OUT"
ARGS=(--dataset "$DATASET" --url "$URL" --speed "$SPEED" --max-chunks "$CHUNKS"
      --timeout "$INFER_TIMEOUT" --duration-s 0
      --image-format jpeg --output "$OUT/chunks.jsonl" "${FLAGS[@]}")
# Validate arguments, dataset and the actual server contract before inference or
# constructing a robot-command publisher. Preserve the health reply with the run.
.venv/bin/python - "$OUT/health.json" "${ARGS[@]}" <<'PY'
import asyncio
import json
import sys
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import aiohttp

from franka_duo_tele_data.delta14_client import ACTION_REPRESENTATION
from franka_duo_tele_data.rgb20d_io import RGB20DContract
from franka_duo_tele_data.smolvla_stream import build_parser, parse_stream_args

args = parse_stream_args(build_parser(), sys.argv[2:])
RGB20DContract(args.dataset)
url = urlsplit(args.url)
if url.scheme not in ('ws', 'wss') or url.path != '/infer':
    raise SystemExit('FASTWAM_URL must be a ws:// or wss:// /infer endpoint')
health_url = urlunsplit(('https' if url.scheme == 'wss' else 'http', url.netloc, '/health', '', ''))

async def check():
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=args.timeout)) as session:
        async with session.get(health_url) as response:
            response.raise_for_status()
            health = await response.json()
    Path(sys.argv[1]).write_text(json.dumps(health, indent=2, allow_nan=False))
    expected = dict(action_dim=14, state_dim=20, horizon=32, action_rate_hz=30,
                    action_representation=ACTION_REPRESENTATION)
    if health.get('ready') is not True or health.get('busy') is not False:
        raise SystemExit('FastWAM is not ready or is busy; retry after it becomes idle')
    if health.get('normalized') is not False or any(health.get(k) != v for k, v in expected.items()):
        raise SystemExit('FastWAM health contract mismatch; inspect saved health.json')
    print(f"FastWAM ready; action rate={30 * args.speed:g} Hz; publish={args.publish}", flush=True)

asyncio.run(check())
PY
echo "Run diagnostics: $PWD/$OUT (JSONL, health and console); MCAP recording disabled"
.venv/bin/python -m franka_duo_tele_data.delta14_stream "${ARGS[@]}" 2>&1 | tee "$OUT/console.log"
