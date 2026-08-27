#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -z "${ROS_DISTRO:-}" ]]; then
  if [[ -f "${HOME}/tmr_env.sh" ]]; then
    # shellcheck disable=SC1090
    set +u
    source "${HOME}/tmr_env.sh"
    set -u
  elif [[ -f /opt/ros/jazzy/setup.bash ]]; then
    # shellcheck disable=SC1091
    set +u
    source /opt/ros/jazzy/setup.bash
    set -u
  fi
fi

CONFIG="${FRANKA_MCAP_CONFIG:-${REPO_ROOT}/configs/tmr_mcap.yaml}"
UV_BIN="${UV_BIN:-}"
if [[ -z "${UV_BIN}" ]]; then
  UV_BIN="$(command -v uv || true)"
fi
if [[ -z "${UV_BIN}" && -x "${HOME}/.local/bin/uv" ]]; then
  UV_BIN="${HOME}/.local/bin/uv"
fi
if [[ -z "${UV_BIN}" ]]; then
  echo "uv was not found; install it or set UV_BIN=/path/to/uv" >&2
  exit 127
fi
cd "${REPO_ROOT}"
exec "${UV_BIN}" run --project "${REPO_ROOT}" --no-sync --frozen --no-default-groups --extra record franka-duo-mcap-record \
  --config "${CONFIG}" --rewarded "$@"
