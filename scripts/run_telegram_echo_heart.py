"""Legacy Telegram harness for manual testing."""

from __future__ import annotations

import logging
import time

from alphonse.agent.main import load_env
from alphonse.extremities.telegram_extremity import build_telegram_extremity_from_env


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    load_env()
    telegram_extremity = build_telegram_extremity_from_env()
    if not telegram_extremity:
        raise RuntimeError("ALPHONSE_ENABLE_TELEGRAM must be true to run this script")

    telegram_extremity.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        telegram_extremity.stop()


if __name__ == "__main__":
    main()
