"""Run Home Assistant domotics smoke test.

Usage:
  python -m alphonse.run_homeassistant_smoke --print-instructions
  python -m alphonse.run_homeassistant_smoke --entity-id input_boolean.alphonse_test_toggle
"""

from alphonse.integrations.homeassistant.smoke_toggle import main


if __name__ == "__main__":
    main()
