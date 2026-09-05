#!/usr/bin/env bash
set -euo pipefail
root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pid_file="${root_dir}/localization.pid"
if [[ ! -f "${pid_file}" ]]; then echo "localization is not managed here"; exit 0; fi
pid="$(cat "${pid_file}")"
if kill -0 "${pid}" 2>/dev/null; then kill -TERM -- "-${pid}"; fi
mv "${pid_file}" "${pid_file}.stopped"
echo "localization process group ${pid} stopped"
