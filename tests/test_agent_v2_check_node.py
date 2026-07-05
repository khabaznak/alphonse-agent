from __future__ import annotations

from alphonse.agent_v2.core.core import CoreLoopContext
from alphonse.agent_v2.core.intelligence.pdca.nodes import check_node
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.messages import CommunicationChannel
from alphonse.agent_v2.core.messages import InMemoryMessageQueue


def test_check_node_marks_new_task_when_acceptance_criteria_are_empty() -> None:
    task = TaskState(goal="Write a file")

    result = check_node(task)

    assert result is task
    assert task.check_verdict == "new"
    assert task.check_new_message_count == 0


def test_check_node_marks_existing_task_without_steering_as_wip() -> None:
    task = TaskState(goal="Continue task", acceptance_criteria_md="- File exists")

    check_node(task, context=CoreLoopContext(messages=InMemoryMessageQueue()))

    assert task.check_verdict == "wip"
    assert task.check_new_message_count == 0


def test_check_node_consumes_same_user_same_project_steering() -> None:
    queue = InMemoryMessageQueue()
    channel = CommunicationChannel(queue)
    channel.queue_message(prompt="Add tests too", user="alex", project_id="alpha")
    channel.queue_message(prompt="Different project", user="alex", project_id="beta")
    task = TaskState(
        goal="Continue task",
        user="alex",
        project_id="alpha",
        acceptance_criteria_md="- Feature works",
    )

    check_node(task, context=CoreLoopContext(messages=queue))

    assert task.check_verdict == "steer"
    assert task.check_new_message_count == 1
    assert '- alex: "Add tests too"' in task.recent_conversation_md
    assert "Different project" not in task.recent_conversation_md
    assert queue.size() == 1


def test_check_node_consumes_same_correlation_id_from_other_user() -> None:
    queue = InMemoryMessageQueue()
    channel = CommunicationChannel(queue)
    channel.queue_message(prompt="No coffee today", user="Gaby", project_id="home", correlation_id="coffee-1")
    task = TaskState(
        goal="Ask Gaby about coffee",
        user="Alex",
        project_id="home",
        correlation_id="coffee-1",
        acceptance_criteria_md="- Answer Alex",
    )

    check_node(task, context=CoreLoopContext(messages=queue))

    assert task.check_verdict == "steer"
    assert task.check_new_message_count == 1
    assert '- Gaby: "No coffee today"' in task.recent_conversation_md
    assert queue.size() == 0


def test_check_node_keeps_new_verdict_after_consuming_steering() -> None:
    queue = InMemoryMessageQueue()
    CommunicationChannel(queue).queue_message(prompt="Add this detail", user="alex", project_id="alpha")
    task = TaskState(goal="New task", user="alex", project_id="alpha")

    check_node(task, context=CoreLoopContext(messages=queue))

    assert task.check_verdict == "new"
    assert task.check_new_message_count == 1
    assert '- alex: "Add this detail"' in task.recent_conversation_md
    assert queue.size() == 0


def test_check_node_does_not_convert_consumed_messages_into_task_states(monkeypatch) -> None:
    queue = InMemoryMessageQueue()
    channel = CommunicationChannel(queue)
    channel.queue_message(prompt="Steering", user="alex", project_id="alpha")
    task = TaskState(
        goal="Existing task",
        user="alex",
        project_id="alpha",
        acceptance_criteria_md="- Done",
    )

    def fail_from_queued_message(*args: object, **kwargs: object) -> TaskState:
        _ = args
        _ = kwargs
        raise AssertionError("steering messages should not become TaskState objects")

    monkeypatch.setattr(TaskState, "from_queued_message", fail_from_queued_message)

    check_node(task, context=CoreLoopContext(messages=queue))

    assert task.check_verdict == "steer"
