from __future__ import annotations

from datetime import datetime
from datetime import timezone

import pytest

from alphonse.agent_v2.core.messages import CommunicationChannel
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.messages import MessageSelector


def test_queue_message_requires_project_provenance() -> None:
    queue = InMemoryMessageQueue()
    channel = CommunicationChannel(queue)

    queued = channel.queue_message(prompt="hello", user="alex", project_id="home")

    assert queued.message.prompt == "hello"
    assert queued.message.user == "alex"
    assert queue.size() == 1


def test_queue_message_rejects_blank_project_id() -> None:
    with pytest.raises(ValueError, match="project_id_required"):
        CommunicationChannel(InMemoryMessageQueue()).queue_message(prompt="hello", user="alex")


def test_queue_message_generates_timezone_aware_local_timestamp() -> None:
    queued = CommunicationChannel(InMemoryMessageQueue()).queue_message(prompt="hello", user="alex", project_id="home")

    assert queued.message.timestamp.tzinfo is not None
    assert queued.message.timestamp.utcoffset() is not None


def test_queue_message_preserves_explicit_timestamp() -> None:
    timestamp = datetime(2026, 7, 4, 12, 30, tzinfo=timezone.utc)

    queued = CommunicationChannel(InMemoryMessageQueue()).queue_message(
        prompt="hello",
        user="alex",
        project_id="home",
        timestamp=timestamp,
    )

    assert queued.message.timestamp == timestamp


def test_selector_dequeues_by_user_project_id_and_tag() -> None:
    queue = InMemoryMessageQueue()
    channel = CommunicationChannel(queue)
    channel.queue_message(prompt="one", user="alex", project_id="alpha", tag="work")
    channel.queue_message(prompt="two", user="gaby", project_id="home", tag="family")

    selected = queue.dequeue(MessageSelector(user="gaby", project_id="home", tag="family"))

    assert selected is not None
    assert selected.message.prompt == "two"


def test_queue_message_preserves_correlation_id_for_selector_lookup() -> None:
    queue = InMemoryMessageQueue()
    channel = CommunicationChannel(queue)
    channel.queue_message(prompt="one", user="alex", project_id="home", correlation_id="task-1")
    channel.queue_message(prompt="two", user="gaby", project_id="home", correlation_id="task-2")

    selected = queue.dequeue(MessageSelector(correlation_id="task-2"))

    assert selected is not None
    assert selected.message.prompt == "two"
    assert selected.message.correlation_id == "task-2"


def test_slash_command_detected_only_at_prompt_start() -> None:
    channel = CommunicationChannel(InMemoryMessageQueue())

    command = channel.queue_message(prompt="/project new", user="alex", project_id="home").message
    indented = channel.queue_message(prompt=" /project new", user="alex", project_id="home").message
    sentence = channel.queue_message(prompt="please /project new", user="alex", project_id="home").message

    assert command.metadata["is_command"] is True
    assert command.metadata["command"] == "project"
    assert command.metadata["command_args"] == "new"
    assert command.metadata["channel"]["integration_id"] == "tui"
    assert indented.metadata["channel"]["channel_target"] == "alex"
    assert sentence.metadata["channel"]["alphonse_user_id"] == "alex"
    assert {key: indented.metadata[key] for key in ("is_command", "command", "command_args")} == {
        "is_command": False,
        "command": "",
        "command_args": "",
    }
    assert {key: sentence.metadata[key] for key in ("is_command", "command", "command_args")} == {
        "is_command": False,
        "command": "",
        "command_args": "",
    }


def test_queue_message_accepts_explicit_channel_metadata() -> None:
    message = CommunicationChannel(InMemoryMessageQueue()).queue_message(
        prompt="hello",
        user="u-alex",
        project_id="home",
        integration_id="telegram-home",
        provider_key="telegram",
        provider_user_id="123",
        channel_target="chat-1",
        provider_message_id="msg-9",
        reply_to_provider_message_id="msg-8",
        thread_id="thread-1",
    ).message

    assert message.metadata["channel"] == {
        "integration_id": "telegram-home",
        "provider_key": "telegram",
        "channel_target": "chat-1",
        "alphonse_user_id": "u-alex",
        "provider_user_id": "123",
        "provider_message_id": "msg-9",
        "reply_to_provider_message_id": "msg-8",
        "thread_id": "thread-1",
    }


def test_legacy_slash_command_metadata_shape_is_still_available() -> None:
    channel = CommunicationChannel(InMemoryMessageQueue())
    command = channel.queue_message(prompt="/project new", user="alex", project_id="home").message

    assert {key: command.metadata[key] for key in ("is_command", "command", "command_args")} == {
        "is_command": True,
        "command": "project",
        "command_args": "new",
    }


def test_blank_prompt_or_user_raises_value_error() -> None:
    channel = CommunicationChannel(InMemoryMessageQueue())

    with pytest.raises(ValueError, match="prompt_required"):
        channel.queue_message(prompt=" ", user="alex", project_id="home")
    with pytest.raises(ValueError, match="user_required"):
        channel.queue_message(prompt="hello", user="", project_id="home")
