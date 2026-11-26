# Evidence review Critic fixture

From the repository root, run
`PYTHONPATH=. .venv/bin/python desktop/tests/critic-action-digest/generate.py`
to generate the disposable result envelopes. With Vite running on port 1420,
open `http://localhost:1420/tests/critic-action-digest/`.

The fixture runs the production A2UI projectors, decoder, React components and
styles with synthetic facts. Select a numbered citation on the Gmail card,
then check focus and the highlighted excerpt in the visible source panel. Use **Compact conversation**
and **Empty results** to change the viewport within the fixture and the data
state. This isolated fixture exercises the renderer without daemon IPC or live
provider content.

## Visual gate

Keep the approved `output/alphonse-gmail-calendar-mocks.html` in place when
generating fixtures. The parity page uses the Evidence review cards from that
file, with the same Lucide glyphs as the app. It gives both cards the same
content and width and captures the background, spacing and both cards in the
panel. The candidate uses the production Desktop message component.

With Chromium open in `playwright-cli` and Vite serving port 1420, run from
the repository root. The comparator requires Pillow.

```sh
playwright-cli run-code --filename=desktop/tests/critic-action-digest/capture-parity.js
python desktop/tests/critic-action-digest/compare_parity.py <new-capture-directory>
```

Review all 16 pairs across light and dark themes and 350px and 670px card
widths. Compare the full panel, Gmail, Calendar, and Gmail with source 8
selected. Every pair must have identical dimensions and zero differing RGBA
pixels. Check keyboard citation activation and focus separately. Repeat the
matched captures in the macOS Tauri WebView and check a regular Gmail summary
through the real Desktop pickup path using disposable provider data. Track
each environment's result independently.
