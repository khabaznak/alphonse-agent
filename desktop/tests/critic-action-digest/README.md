# Action digest Critic fixture

From the repository root, run
`PYTHONPATH=. .venv/bin/python desktop/tests/critic-action-digest/generate.py`
to generate the disposable result envelopes. With Vite running on port 1420,
open `http://localhost:1420/tests/critic-action-digest/`.

The fixture runs the production A2UI projectors, decoder, React components and
styles with synthetic facts. Select a numbered citation on the Gmail card,
then check focus and the highlighted excerpt. Use **Compact conversation**
and **Empty results** to change the viewport within the fixture and the data
state. This isolated fixture exercises the renderer without daemon IPC or live
provider content.
