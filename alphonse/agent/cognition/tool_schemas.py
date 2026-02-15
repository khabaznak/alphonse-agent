from __future__ import annotations

from typing import Any


def planner_tool_schemas() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "askQuestion",
                "description": "Ask the user one clear question and wait for their answer.",
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
                "name": "createReminder",
                "description": "Create a reminder for someone at a specific time.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "ForWhom": {"type": "string"},
                        "Time": {"type": "string"},
                        "Message": {"type": "string"},
                    },
                    "required": ["ForWhom", "Time", "Message"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "local_audio_output.speak",
                "description": "Speak text out loud on the local computer using OS-native TTS.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "voice": {"type": "string"},
                        "blocking": {"type": "boolean"},
                        "volume": {"type": "number"},
                    },
                    "required": ["text"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "stt_transcribe",
                "description": "Transcribe an audio asset by asset_id into text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "asset_id": {"type": "string"},
                        "language_hint": {"type": "string"},
                    },
                    "required": ["asset_id"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "telegramGetFileMeta",
                "description": "Resolve Telegram file metadata from a file_id.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_id": {"type": "string"},
                    },
                    "required": ["file_id"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "telegramDownloadFile",
                "description": "Download a Telegram file by file_id and return local path details.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_id": {"type": "string"},
                        "sandbox_alias": {"type": "string"},
                        "relative_path": {"type": "string"},
                    },
                    "required": ["file_id"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "transcribeTelegramAudio",
                "description": "Download Telegram audio by file_id and transcribe it to text.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_id": {"type": "string"},
                        "language": {"type": "string"},
                        "sandbox_alias": {"type": "string"},
                    },
                    "required": ["file_id"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "analyzeTelegramImage",
                "description": "Download Telegram image by file_id and analyze it with a prompt.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_id": {"type": "string"},
                        "prompt": {"type": "string"},
                        "sandbox_alias": {"type": "string"},
                    },
                    "required": ["file_id"],
                    "additionalProperties": False,
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "getMySettings",
                "description": "Get runtime settings for current conversation context.",
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
                "name": "python_subprocess",
                "description": "Execute a Python subprocess command on the local system. Use to install missing tools or for other system-level operations. Be cautious with this tool and avoid running untrusted commands.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout_seconds": {"type": "number"},
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
            },
        },
    ]
