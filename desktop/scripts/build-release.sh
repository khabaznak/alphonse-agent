#!/usr/bin/env zsh
set -euo pipefail

root="$(cd "$(dirname "$0")/../.." && pwd)"
target="$(rustc -vV | awk '/^host:/ { print $2 }')"
sidecar="$root/desktop/src-tauri/binaries/alphonse-daemon-$target"
backup="$(mktemp -t alphonse-daemon-sidecar)"
had_sidecar=false

if [[ -f "$sidecar" ]]; then
  cp "$sidecar" "$backup"
  had_sidecar=true
fi

restore_sidecar() {
  if [[ "$had_sidecar" == true ]]; then
    cp "$backup" "$sidecar"
  else
    rm -f "$sidecar"
  fi
  rm -f "$backup"
}
trap restore_sidecar EXIT

zsh "$root/desktop/scripts/build-daemon-sidecar.sh" "$target"
cd "$root/desktop"
"$root/desktop/node_modules/.bin/tauri" build --target "$target" "$@"
