# Alphonse Desktop

The macOS-first native client for the Alphonse v2 daemon. The interface is a
Tauri 2 shell with a React/TypeScript frontend; all agent state and processing
remain in the Python daemon.

## Development

```bash
npm install
npm run tauri dev
```

Desktop first connects to the configured local v2 socket. If no daemon is
available, development builds launch the repository's `.venv/bin/python -m
alphonse.agent_v2.daemon` from the project root, falling back to `python3` when
no project virtual environment exists, then wait for it to become healthy.
Install project dependencies first (`.venv/bin/python -m pip install -r
requirements.txt`), including the event-worker JSON Schema dependency.

## Release sidecar

Tauri expects a target-specific executable at
`src-tauri/binaries/alphonse-daemon-<target-triple>`. The committed macOS ARM
wrapper starts the daemon from the project virtual environment for development. Replace it with the
standalone executable before packaging:

```bash
python -m pip install pyinstaller
./scripts/build-daemon-sidecar.sh aarch64-apple-darwin
npm run tauri build
```

The packaged executable is launched only when the app cannot connect to an
already-running daemon. Closing Desktop leaves the daemon running.
