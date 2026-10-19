from types import SimpleNamespace

from alphonse.agent_v2.core.core import CoreLoopContext, ProcessingStatus
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.intelligence.v3.processor import HierarchicalCAPDProcessor
from alphonse.agent_v2.core.messages import InMemoryMessageQueue
from alphonse.agent_v2.core.questions import SQLiteQuestionStore
from alphonse.agent_v2.daemon import V2Daemon
from alphonse.agent_v2.runtime import build_runtime_host


REASON = "The task stopped after repeated phases made no acceptance progress."
EXPLANATION = "No pude verificar la temperatura: el Python que ejecutó el cliente no tiene instalado requests. Hay que usar el entorno del artifact con sus dependencias."


class FailureInference:
    def __init__(self, *, fail=False):
        self.requests = []
        self.fail = fail

    def generate_markdown(self, request):
        self.requests.append(request)
        if self.fail:
            raise RuntimeError("backend unavailable")
        return SimpleNamespace(content=EXPLANATION)


def _task():
    return TaskState(
        task_id="current", message_id="current", user="alex",
        goal="Puedes revisar la temperatura del estudio?", status="failed",
        outcome={"status": "failure", "reason": REASON},
        metadata={"v3_no_progress_stop": True},
        hierarchical_state={"evidence": {"entries": [{
            "evidence_ref": "action:last", "tool_id": "native.bash", "status": "failed",
            "arguments": {"command": "do not include command or secret value"},
            "error": "ModuleNotFoundError: No module named 'requests'",
            "result": {"exit_code": 1, "stderr": "ModuleNotFoundError: No module named 'requests'"},
        }]}},
    )


def test_no_progress_failure_keeps_failed_status_and_prepares_evidence_based_explanation():
    task = _task()
    inference = FailureInference()
    questions = SQLiteQuestionStore(":memory:")
    context = CoreLoopContext(messages=InMemoryMessageQueue(), inference=inference, question_store=questions)

    result = HierarchicalCAPDProcessor._result(task, context)

    assert result.status == ProcessingStatus.FAILED
    assert result.error == f"v3_task_failed:{REASON}"
    assert task.outcome == {"status": "failure", "reason": REASON}
    assert task.metadata["v3_failure_response"]["message"] == EXPLANATION
    request = inference.requests[0]
    assert request.tools == ()
    assert "requests" in request.prompt
    assert task.goal in request.prompt
    assert "do not include command or secret value" not in request.prompt
    assert questions.load_task_checkpoint("current").metadata["v3_failure_response"]["message"] == EXPLANATION
    HierarchicalCAPDProcessor._result(task, context)
    assert len(inference.requests) == 1


def test_failure_explanation_generation_failure_preserves_original_failure():
    task = _task()
    context = CoreLoopContext(messages=InMemoryMessageQueue(), inference=FailureInference(fail=True))

    result = HierarchicalCAPDProcessor._result(task, context)

    assert result.status == ProcessingStatus.FAILED
    assert result.error == f"v3_task_failed:{REASON}"
    assert task.metadata["v3_failure_response"]["source"] == "v3_failure_fallback"
    assert "not been fully verified" in task.metadata["v3_failure_response"]["message"]
    assert task.metadata["v3_failure_response_error_type"] == "RuntimeError"


def test_failure_explanation_keeps_successful_observations_and_redacts_named_secrets():
    task = _task()
    task.hierarchical_state["evidence"]["entries"].insert(0, {
        "evidence_ref": "action:earlier", "tool_id": "native.bash", "status": "success",
        "result": {"exit_code": 0, "stdout": '{"temperature_c":24}', "api_key": "DO_NOT_FORWARD"},
    })
    inference = FailureInference()
    HierarchicalCAPDProcessor._result(task, CoreLoopContext(messages=InMemoryMessageQueue(), inference=inference))
    assert "temperature_c" in inference.requests[0].prompt and "24" in inference.requests[0].prompt
    assert "DO_NOT_FORWARD" not in inference.requests[0].prompt
    assert task.status == "failed"


def test_failure_closure_reuses_prepared_response_without_another_model_call():
    task = _task()
    task.metadata["act_directive"] = {"closure_only": True}
    task.metadata["prepared_user_response"] = {"source": "native.respond", "message": EXPLANATION}
    inference = FailureInference()

    HierarchicalCAPDProcessor._result(task, CoreLoopContext(messages=InMemoryMessageQueue(), inference=inference))

    assert task.metadata["v3_failure_response"]["message"] == EXPLANATION
    assert inference.requests == []


def test_daemon_delivers_failure_explanation_to_origin_and_records_actual_text():
    runtime = build_runtime_host()
    queued = runtime.channel.queue_message(
        prompt="Puedes revisar la temperatura del estudio?", user="alex", project_id="home",
        integration_id="desktop", provider_key="desktop", channel_target="alex",
    )
    task = _task()
    task.message_id = queued.message_id
    task.metadata["v3_failure_response"] = {"source": "v3_terminal_failure", "message": EXPLANATION}

    V2Daemon(runtime)._notify_inbound_failure(queued, error=f"v3_task_failed:{REASON}", task_state=task.to_dict())

    outbound = runtime.outbox.list()[0]
    assert outbound.kind == "task_failed"
    assert outbound.integration_id == "desktop"
    assert outbound.message == EXPLANATION
    events = runtime.conversation_store.list_recent(owner_user_id="alex")
    assert events[-1].content == EXPLANATION


def test_daemon_does_not_reuse_explanation_from_a_different_task():
    runtime = build_runtime_host()
    queued = runtime.channel.queue_message(
        prompt="Current request", user="alex", integration_id="desktop",
        provider_key="desktop", channel_target="alex",
    )
    task = _task()
    task.metadata["v3_failure_response"] = {"message": EXPLANATION}

    V2Daemon(runtime)._notify_inbound_failure(queued, error=f"v3_task_failed:{REASON}", task_state=task.to_dict())

    message = runtime.outbox.list()[0].message
    assert message != EXPLANATION
    assert "rephrase" not in message


def test_daemon_terminal_failure_projects_explanation_instead_of_generic_fallback():
    inference = FailureInference()
    runtime = build_runtime_host()

    class FailedProcessor:
        def process(self, task, context):
            failed = _task()
            task.status = failed.status
            task.outcome = failed.outcome
            task.hierarchical_state = failed.hierarchical_state
            context.inference = inference
            return HierarchicalCAPDProcessor._result(task, context)

    runtime.core.intelligence = FailedProcessor()
    runtime.core.memory = None
    runtime.channel.queue_message(
        prompt="Puedes revisar la temperatura del estudio?", user="alex",
        integration_id="desktop", provider_key="desktop", channel_target="alex",
    )

    step = V2Daemon(runtime).run_once()

    assert step.status.value == "failed"
    notifications = [item for item in runtime.outbox.list() if item.kind == "task_failed"]
    assert len(notifications) == 1
    assert notifications[0].message == EXPLANATION


def test_required_service_failure_does_not_attempt_another_inference():
    task = _task()
    task.outcome["reason"] = "system_one_unavailable:task_admission"
    inference = FailureInference()

    result = HierarchicalCAPDProcessor._result(task, CoreLoopContext(messages=InMemoryMessageQueue(), inference=inference))

    assert result.error == task.outcome["reason"]
    assert inference.requests == []
