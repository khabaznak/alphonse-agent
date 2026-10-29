# Commands

Use native.bash with runtime set to the documented available Python interpreter,
skill_id `skill:ac-fixture`, binding_names `[settings, device]`, and commands:

- `"$ALPHONSE_SKILL_RUNTIME" scripts/ac.py temperature --room study`
- `"$ALPHONSE_SKILL_RUNTIME" scripts/ac.py set --room study --target 23`
- `"$ALPHONSE_SKILL_RUNTIME" scripts/ac.py verify --room study --target 23`

`set` acknowledges a request; only `verify` proves the observed target matches.
All commands emit JSON. Errors exit nonzero (2), including state_mismatch.
Use at most one write per requested change, then readback; failures require
reconciliation, not blind retry. This fixture uses only Python's standard library.
