from __future__ import annotations

from typing import Any


def planner_tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "askQuestion",
                "description": "Ask the user one clear question and wait for their answer.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                    },
                    "required": ["question"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "getTime",
                "description": "Get your current time now.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "createTimeEventTrigger",
                "description": "Create a time-based trigger from a time expression.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "time": {"type": "string"},
                    },
                    "required": ["time"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "scheduleReminder",
                "description": "Schedule a reminder using a trigger.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "Message": {"type": "string"},
                        "To": {"type": "string"},
                        "From": {"type": "string"},
                        "EventTrigger": {
                            "type": "object",
                            "properties": {
                                "type": {"type": "string"},
                                "time": {"type": "string"},
                            },
                            "required": ["type", "time"],
                            "additionalProperties": True,
                        },
                    },
                    "required": ["Message", "To", "From", "EventTrigger"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "getMySettings",
                "description": "Get runtime settings for current conversation context.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "getUserDetails",
                "description": "Get known user and channel details for current conversation context.",
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
            },
        },
    ]
