#!/usr/bin/env bash
# Offline archive inspection/playback. No ROS or robot publishers.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec "${LABS_PYTHON:-.venv/bin/python}" -m franka_duo_tele_data.labs_action_recording "$@"
