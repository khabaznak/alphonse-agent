# Home Assistant Smoke Test

This smoke flow validates the domotics wiring using:
1. `domotics.query`
2. `domotics.subscribe` (10s window)
3. `domotics.execute` (toggle on/off)

## 1) Create test helper in Home Assistant

Option A (UI):
- `Settings` -> `Devices & Services` -> `Helpers` -> `Create Helper` -> `Toggle`
- Name: `Alphonse Test Toggle`
- Ensure entity id is `input_boolean.alphonse_test_toggle`

Option B (`configuration.yaml`):

```yaml
input_boolean:
  alphonse_test_toggle:
    name: Alphonse Test Toggle
```

Restart Home Assistant after YAML changes.

## 2) Ensure Alphonse env is configured

In `alphonse/agent/.env` set:

```bash
HA_BASE_URL=http://homeassistant.local:8123
HA_TOKEN=your-long-lived-token
```

## 3) Run smoke helper

```bash
python -m alphonse.run_homeassistant_smoke --print-instructions
python -m alphonse.run_homeassistant_smoke --entity-id input_boolean.alphonse_test_toggle --duration-seconds 10
```

The command prints JSON with `before`, `subscribe`, `execute_on`, `execute_off`, and `after` payloads.
