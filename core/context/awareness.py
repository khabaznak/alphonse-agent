from core.context.providers.time_provider import get_time_context
from core.context.providers.system_provider import get_system_context

def get_awareness_snapshot():
    return {
        "time": get_time_context(),
        "system": get_system_context()
    }
