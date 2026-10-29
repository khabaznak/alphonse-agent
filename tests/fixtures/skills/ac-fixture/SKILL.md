---
name: ac-fixture
description: Read room temperature and verify requested AC setpoints on a disposable synthetic device. Use for room temperature and AC control acceptance tests only.
compatibility: Python 3 standard library; configured settings and device data bindings. No network or real devices.
---
# Synthetic room climate

Read [usage](references/usage.md) and [device resolution](references/devices.md)
before composing commands. Execute the ordinary scripts/ac.py CLI with the
configured Python runtime through native Bash, skill_id skill:ac-fixture and
binding_names [settings, device]. No project or artifact registration is needed.
Bindings, not guessed paths or room names, supply configuration and live state.

A temperature read must run now. Report value, units, device source and observation
time from its JSON output. A setpoint is not a room temperature.
Only change the explicitly requested room/setpoint under authorized effects.
Run a separate verification after a change; a command acknowledgement alone is
not verified success. Stop on ambiguous_room, unknown_room or device_unavailable
and explain the error without inventing a reading or claiming a change.
After interruption read back first; do not repeat a completed or uncertain write.
Installation and skill selection alone are never acceptance evidence.
