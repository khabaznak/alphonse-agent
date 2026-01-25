from core.context.awareness import get_awareness_snapshot
from rex.cognition.providers.localai import LocalAIClient


STATUS_SYSTEM_PROMPT = (
    "You are Rex, a calm and restrained domestic presence.\n\n"
    "Given the following system snapshot, produce a short, neutral status message.\n"
    "Do not suggest actions.\n"
    "Do not speculate.\n"
    "Keep it under 2 sentences."
)


def reason_about_status():
    snapshot = get_awareness_snapshot()
    client = LocalAIClient()

    message = client.complete(
        system_prompt=STATUS_SYSTEM_PROMPT,
        user_prompt=str(snapshot),
    )

    return message, snapshot
