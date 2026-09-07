#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
if [[ -f "${HOME}/tmr_env.sh" ]]; then
  set +u
  source "${HOME}/tmr_env.sh"
  set -u
fi
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
# Use the host's existing uv environment; no dependency resolution on a live run.
exec "${REPO_ROOT}/.venv/bin/python" -m franka_duo_tele_data.replay_rgb20d "$@"
