#!/usr/bin/env bash
# ZED head-camera extrinsic calibration (PnP on a bead held in the gripper).
# Read-only with respect to the robot: it only subscribes to current_pose and the
# ZED image/camera_info. Move the arms by hand (guiding mode) with the impedance
# controllers inactive.
#
# Collect:  bash scripts/run_zed_pnp_calib.sh --tool-offset-m 0.045 [--arm left] [--resume]
# Re-solve: bash scripts/run_zed_pnp_calib.sh --solve outputs/zed_pnp/samples.json
# Check:    bash scripts/run_zed_pnp_calib.sh --tool-offset-m 0.045 --check configs/zed_pnp_calibration.json
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
MANIFEST="${ZED_PNP_MANIFEST:-datasets/franka_duo_lerobot_rgb20d_v1/franka_duo_extras/derived_manifest.json}"
exec "${REPO_ROOT}/.venv/bin/python" -m franka_duo_tele_data.zed_pnp_calib --manifest "${MANIFEST}" "$@"
