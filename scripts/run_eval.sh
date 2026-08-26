#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ROS and camera drivers stay on the host; this script only starts the Python
# evaluator and never creates a container.
if [[ -z "${ROS_DISTRO:-}" ]]; then
  if [[ -f "${HOME}/tmr_env.sh" ]]; then
    # shellcheck disable=SC1090
    source "${HOME}/tmr_env.sh"
  elif [[ -f /opt/ros/jazzy/setup.bash ]]; then
    # shellcheck disable=SC1091
    source /opt/ros/jazzy/setup.bash
  fi
fi

CONFIG="${FRANKA_EVAL_CONFIG:-${REPO_ROOT}/configs/tmr_eval.yaml}"
if [[ $# -lt 1 ]]; then
  echo "usage: $0 /path/to/bundle [eval options...]" >&2
  echo "   or: $0 --checkpoint /path/to/checkpoint --manifest /path/to/manifest.json [options...]" >&2
  echo "eval always records one raw MCAP episode; use --once or --max-steps for a complete bag" >&2
  exit 2
fi

UV_EXTRAS=(--extra eval)
if [[ "${FRANKA_LEROBOT_POLICY:-0}" == "1" ]]; then
  UV_EXTRAS+=(--extra lerobot-policy)
fi

MCAP_ARGS=()
if [[ -n "${FRANKA_MCAP_CONFIG:-}" ]]; then
  MCAP_ARGS+=(--mcap-config "${FRANKA_MCAP_CONFIG}")
fi

cd "${REPO_ROOT}"
if [[ "$1" == "--help" || "$1" == "-h" ]]; then
  exec uv run --project "${REPO_ROOT}" "${UV_EXTRAS[@]}" franka-duo-eval --help
fi
if [[ "$1" == "--bundle" || "$1" == "--checkpoint" ]]; then
  exec uv run --project "${REPO_ROOT}" "${UV_EXTRAS[@]}" franka-duo-eval \
    --config "${CONFIG}" "${MCAP_ARGS[@]}" "$@"
fi

BUNDLE="$1"
shift
exec uv run --project "${REPO_ROOT}" "${UV_EXTRAS[@]}" franka-duo-eval \
  --bundle "${BUNDLE}" --config "${CONFIG}" "${MCAP_ARGS[@]}" "$@"
