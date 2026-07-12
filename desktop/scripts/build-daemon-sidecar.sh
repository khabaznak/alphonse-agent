#!/usr/bin/env zsh
set -euo pipefail

# Builds the standalone daemon executable expected by Tauri's externalBin
# setting. Run this once per macOS target triple before `npm run tauri build`.
target="${1:?usage: build-daemon-sidecar.sh <rust-target-triple>}"
root="$(cd "$(dirname "$0")/../.." && pwd)"
output="$root/desktop/src-tauri/binaries/alphonse-daemon-$target"

cd "$root"
python -m PyInstaller --onefile --name "alphonse-daemon-$target" --distpath "$(dirname "$output")" --workpath /tmp/alphonse-desktop-pyinstaller --specpath /tmp alphonse/agent_v2/daemon.py
