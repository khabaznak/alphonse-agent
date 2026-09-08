#!/usr/bin/env zsh
set -euo pipefail

# Builds the standalone daemon executable expected by Tauri's externalBin
# setting. The current Rust host target is used unless one is supplied.
target="${1:-$(rustc -vV | awk '/^host:/ { print $2 }')}"
if [[ -z "$target" ]]; then
  print -u2 "could not determine Rust target triple"
  exit 1
fi
root="$(cd "$(dirname "$0")/../.." && pwd)"
output="$root/desktop/src-tauri/binaries/alphonse-daemon-$target"
python_bin="$root/.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  python_bin="python"
fi

cd "$root"
"$python_bin" -m PyInstaller --onefile --name "alphonse-daemon-$target" --distpath "$(dirname "$output")" --workpath /tmp/alphonse-desktop-pyinstaller --specpath /tmp alphonse/agent_v2/daemon.py
