def describe_awareness(snapshot: dict) -> str:
    time_ctx = snapshot.get("time", {})
    system_ctx = snapshot.get("system", {})

    day_period = time_ctx.get("day_period", "the day")
    status = system_ctx.get("atrium_status", "unknown")

    if status != "online":
        return "Atrium is not fully available at the moment."

    return (
        f"Atrium is online. "
        f"It is {day_period}, "
        f"and everything is steady."
    )
