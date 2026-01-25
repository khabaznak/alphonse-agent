from datetime import datetime
import time

def get_time_context():
    now = datetime.now().astimezone()

    hour = now.hour

    return {
        "local_time": now.strftime("%H:%M"),
        "day": now.strftime("%A"),
        "date": now.strftime("%Y-%m-%d"),
        "timezone": str(now.tzinfo),
        "iso_timestamp": now.isoformat(),
        "epoch_seconds": int(time.time()),
        "is_night": hour < 6 or hour >= 20,
        "day_period": get_day_period(hour),
    }


def get_day_period(hour: int) -> str:
    if 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 18:
        return "afternoon"
    elif 18 <= hour < 22:
        return "evening"
    else:
        return "night"
