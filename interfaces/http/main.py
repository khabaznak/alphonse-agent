import html

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from rex.cognition.status_reasoning import reason_about_status

app = FastAPI(
    title="Atrium",
    description="Domestic presence interface",
    version="0.1.0"
)

@app.get("/", response_class=HTMLResponse)
def index():
    return """
    <!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Rex — Atrium</title>
        <script src="https://unpkg.com/htmx.org@1.9.12"></script>
        <style>
          :root {
            color-scheme: light;
            --ink: #1d1b18;
            --paper: #f4f0ea;
            --accent: #225b5b;
            --accent-soft: #d7e4e2;
          }
          * { box-sizing: border-box; }
          body {
            margin: 0;
            font-family: "IBM Plex Serif", "Georgia", serif;
            background: radial-gradient(circle at top, #ffffff 0%, var(--paper) 55%, #efe7db 100%);
            color: var(--ink);
          }
          main {
            max-width: 880px;
            margin: 0 auto;
            padding: 64px 24px 96px;
          }
          header {
            display: flex;
            align-items: baseline;
            justify-content: space-between;
            border-bottom: 1px solid rgba(0,0,0,0.08);
            padding-bottom: 20px;
            margin-bottom: 36px;
          }
          h1 {
            font-size: clamp(2.4rem, 5vw, 3.4rem);
            margin: 0;
            letter-spacing: -0.02em;
          }
          .subtitle {
            font-family: "IBM Plex Sans", "Trebuchet MS", sans-serif;
            font-size: 0.95rem;
            letter-spacing: 0.2em;
            text-transform: uppercase;
            color: rgba(0,0,0,0.55);
          }
          .panel {
            background: #fff;
            border-radius: 18px;
            padding: 28px;
            box-shadow: 0 18px 42px rgba(0,0,0,0.08);
            border: 1px solid rgba(0,0,0,0.08);
          }
          .panel h2 {
            margin: 0 0 12px;
            font-size: 1.5rem;
          }
          .status-message {
            font-size: 1.2rem;
            line-height: 1.5;
            margin: 0 0 20px;
          }
          .actions {
            display: flex;
            gap: 12px;
            align-items: center;
            margin-bottom: 20px;
          }
          button {
            border: none;
            background: var(--accent);
            color: #fff;
            padding: 10px 16px;
            border-radius: 999px;
            font-size: 0.95rem;
            cursor: pointer;
            font-family: "IBM Plex Sans", "Trebuchet MS", sans-serif;
          }
          .indicator {
            font-family: "IBM Plex Sans", "Trebuchet MS", sans-serif;
            font-size: 0.85rem;
            color: rgba(0,0,0,0.5);
          }
          dl {
            display: grid;
            grid-template-columns: minmax(140px, 200px) 1fr;
            gap: 10px 16px;
            margin: 0;
            font-family: "IBM Plex Sans", "Trebuchet MS", sans-serif;
            font-size: 0.95rem;
          }
          dt {
            color: rgba(0,0,0,0.6);
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.18em;
          }
          dd {
            margin: 0;
          }
          .soft {
            padding: 16px;
            border-radius: 12px;
            background: var(--accent-soft);
            margin-top: 18px;
          }
          @media (max-width: 640px) {
            header { flex-direction: column; gap: 10px; align-items: flex-start; }
            dl { grid-template-columns: 1fr; }
          }
        </style>
      </head>
      <body>
        <main>
          <header>
            <div>
              <h1>Rex</h1>
              <div class="subtitle">Atrium presence</div>
            </div>
            <div class="subtitle">Minimal UI · HTMX</div>
          </header>

          <section class="panel">
            <h2>Current State</h2>
            <div class="actions">
              <button
                hx-get="/status/fragment"
                hx-target="#status-panel"
                hx-swap="innerHTML"
                hx-indicator="#status-indicator"
              >
                Refresh status
              </button>
              <div id="status-indicator" class="indicator" hidden>Listening…</div>
            </div>
            <div
              id="status-panel"
              hx-get="/status/fragment"
              hx-trigger="load"
              hx-swap="innerHTML"
              hx-indicator="#status-indicator"
            >
              <p class="status-message">Gathering status…</p>
            </div>
          </section>
        </main>
      </body>
    </html>
    """


def _format_snapshot(snapshot: dict) -> str:
    rows = []
    for group_name, group_data in snapshot.items():
        rows.append(
            f"<div class='soft'><strong>{html.escape(group_name.capitalize())}</strong></div>"
        )
        for key, value in group_data.items():
            rows.append(
                "<dt>{}</dt><dd>{}</dd>".format(
                    html.escape(str(key).replace("_", " ")),
                    html.escape(str(value))
                )
            )
    return "<dl>{}</dl>".format("".join(rows))


@app.get("/status/fragment", response_class=HTMLResponse)
def get_status_fragment():
    message, snapshot = reason_about_status()

    return """
      <p class="status-message">{}</p>
      {}
    """.format(
        html.escape(message.strip()),
        _format_snapshot(snapshot),
    )


@app.get("/status")
def get_status():
    message, snapshot = reason_about_status()

    return {
        "message": message,
        "awareness": snapshot
    }
