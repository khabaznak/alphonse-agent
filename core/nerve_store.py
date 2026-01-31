from __future__ import annotations

from typing import Any


def list_signals(limit: int = 200) -> list[dict[str, Any]]:
    return []


def list_states(limit: int = 200) -> list[dict[str, Any]]:
    return []


def list_transitions(limit: int = 200) -> list[dict[str, Any]]:
    return []


def list_signal_queue(limit: int = 200) -> list[dict[str, Any]]:
    return []


def get_signal(signal_id: str) -> dict[str, Any] | None:
    return None


def get_state(state_id: str) -> dict[str, Any] | None:
    return None


def get_transition(transition_id: str) -> dict[str, Any] | None:
    return None


def create_signal(payload: dict[str, Any]) -> dict[str, Any]:
    return payload


def create_state(payload: dict[str, Any]) -> dict[str, Any]:
    return payload


def create_transition(payload: dict[str, Any]) -> dict[str, Any]:
    return payload


def update_signal(signal_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    return payload


def update_state(state_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    return payload


def update_transition(transition_id: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    return payload


def delete_signal(signal_id: str) -> dict[str, Any] | None:
    return None


def delete_state(state_id: str) -> dict[str, Any] | None:
    return None


def delete_transition(transition_id: str) -> dict[str, Any] | None:
    return None
