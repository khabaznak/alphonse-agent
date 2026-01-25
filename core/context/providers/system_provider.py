import time

_START_TIME = time.time()


def get_system_context():
    now = time.time()
    uptime_seconds = int(now - _START_TIME)

    return {
        "atrium_status": "online",
        "uptime_seconds": uptime_seconds,
        "started_at_epoch": int(_START_TIME),
    }
