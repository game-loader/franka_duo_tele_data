#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${ROS_DISTRO:-}" ]]; then
  if [[ -f "${HOME}/tmr_env.sh" ]]; then
    # shellcheck disable=SC1090
    source "${HOME}/tmr_env.sh"
  elif [[ -f /opt/ros/jazzy/setup.bash ]]; then
    # shellcheck disable=SC1091
    source /opt/ros/jazzy/setup.bash
  fi
fi

CONFIG="${FRANKA_MCAP_CONFIG:-${REPO_ROOT}/configs/tmr_mcap.yaml}"
cd "${REPO_ROOT}"
exec uv run --project "${REPO_ROOT}" --extra record franka-duo-mcap-record \
  --config "${CONFIG}" --rewarded "$@"
