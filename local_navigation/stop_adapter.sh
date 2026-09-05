#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pid_file="${root_dir}/adapter.pid"

if [[ ! -f "${pid_file}" ]]; then
  echo "adapter is not managed by this directory"
  exit 0
fi

pid="$(cat "${pid_file}")"
if kill -0 "${pid}" 2>/dev/null; then
  kill -TERM -- "-${pid}"
  echo "adapter process group ${pid} stopped"
else
  echo "adapter PID ${pid} is no longer running"
fi
mv "${pid_file}" "${pid_file}.stopped"
