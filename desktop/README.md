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

## WebKit memory diagnostics

Set `ALPHONSE_DESKTOP_DIAGNOSTIC_MODE` before launching the desktop executable
to isolate recurring work in the WebKit renderer. Supported values are
`static`, `ping-only`, `poll-no-commit`, `render-only`, `history-static`,
`history-render`, `history-render-plain`, `history-render-memo`,
`history-render-timeline-memo`, and `normal`.
Unknown or missing values use `normal`.

`poll-no-commit` performs real Desktop polls but deliberately does not render or
acknowledge their contents. Use it only with an idle daemon that has no queued
messages, active work, questions, or scheduled-task notifications.

The four `history-*` modes load up to 100 real Home conversation messages once.
They respectively leave the history static, rerender its Markdown, rerender it
as plain text, or rerender the parent while memoizing the Markdown bubbles.
Set `ALPHONSE_DESKTOP_DIAGNOSTIC_PROJECT_ID` to load a specific project's
timeline instead of Home.

`history-render-timeline-memo` exercises the production fix: the parent keeps
rerendering while the complete unchanged timeline and its message bubbles retain
their existing React element identities.

After launching a fresh app process, identify its new WebKit WebContent PID and
record the standard one-minute warm-up plus five-minute sample:

```bash
./scripts/sample-webkit-memory.sh WEBKIT_PID MODE
```

The script accepts optional warm-up, duration, interval, and output-file
arguments after the mode. It writes CSV measurements from macOS `vmmap`; the
terminal may request permission to inspect the WebKit process.
