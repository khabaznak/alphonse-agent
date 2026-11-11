"""Opt-in browser workflow foundation; no transport or tool is enabled here.

The embedding runtime supplies an already authorized, task-scoped transport and
live grant/cancellation checks. This service never writes browser payloads to a
checkpoint. It deliberately refuses to recover an uncertain mutation by retrying.
"""
from __future__ import annotations

from threading import RLock
from typing import Any, Callable
from datetime import datetime, timezone

from alphonse.agent_v2.browser_private_data import BrowserPrivateData
from alphonse.agent_v2.core.intelligence.v3 import browser_workflow as workflow


def browser_grant_check(store: Any, actor_active: Callable[[str], bool]) -> Callable[[str, str], str | None]:
    """Adapt the browser access store's current actor grant to a revision token."""
    def check(actor: str, task_id: str) -> str | None:
        if not actor or not task_id or not actor_active(actor):
            return None
        grant = store.grant_snapshot(actor)
        if grant is None:
            return None
        revision, grant_id = grant
        return f"{revision}:{grant_id}"
    return check


class BrowserWorkflowService:
    def __init__(self, *, checkpoint_store: Any, transport: Any,
                 authorized: Callable[[str, str], str | None], cancelled: Callable[[], bool],
                 enabled: bool = False, private_data: BrowserPrivateData | None = None) -> None:
        self.checkpoints = checkpoint_store
        self.transport = transport
        self.authorized = authorized
        self.cancelled = cancelled
        self.enabled = enabled
        self.private_data = private_data or BrowserPrivateData()
        # The live grant callback returns an opaque, revision-specific grant ID.
        # Regranting access must produce a different ID, even for the same actor.
        self._sessions: dict[tuple[str, str], tuple[str, str, str]] = {}
        self._lock = RLock()

    def _check(self, task: Any, *, mutation: bool) -> str:
        try:
            grant = self.authorized(task.user, task.task_id) if task.user and task.task_id else None
        except Exception:
            grant = None
        if (not self.enabled or not task.user or not task.task_id or task.status != "running"
                or task.intelligence_engine != "hierarchical_v3" or self.cancelled()
                or not isinstance(grant, str) or not grant):
            if task.user and task.task_id:
                self._sessions.pop((task.user, task.task_id), None)
                self.private_data.clear_task(actor=task.user, task=task.task_id)
            raise workflow.BrowserWorkflowError("browser_workflow_access_denied")
        tactical = task.hierarchical_state
        phase = tactical.get("phase") if isinstance(tactical, dict) else None
        if not isinstance(phase, dict) or tactical.get("status") != "running" or tactical.get("cancellation_pending"):
            raise workflow.BrowserWorkflowError("browser_workflow_phase_denied")
        subgoal = next((s for s in phase.get("subgoals", []) if isinstance(s, dict)
                        and s.get("subgoal_id") == tactical.get("active_subgoal_id")), None)
        if (not subgoal or "browser" not in subgoal.get("allowed_capabilities", [])
                or (mutation and ("external_irreversible" not in subgoal.get("allowed_side_effects", [])
                    or (phase.get("mutation_scope") or {}).get("allow_external_effects") is not True))):
            raise workflow.BrowserWorkflowError("browser_workflow_phase_denied")
        phase_id = phase.get("phase_id")
        if not workflow._id(phase_id):
            raise workflow.BrowserWorkflowError("browser_workflow_phase_denied")
        return phase_id

    def begin(self, task: Any) -> None:
        with self._lock:
            self._check(task, mutation=False)
            grant = self.authorized(task.user, task.task_id)
            if not isinstance(grant, str) or not grant:
                raise workflow.BrowserWorkflowError("browser_workflow_access_denied")
            identity = self._transport_identity(task)
            state = workflow.create(task)
            self.checkpoints.save_active_browser_checkpoint(task, expected_workflow=None)
            self._sessions[(task.user, task.task_id)] = (state["session"], grant, identity)

    def _transport_identity(self, task: Any) -> str:
        try:
            identity = self.transport.session_identity(task.user, task.task_id)
        except Exception:
            raise workflow.BrowserWorkflowError("browser_workflow_session_unavailable") from None
        if not isinstance(identity, str) or not identity or len(identity) > 256:
            raise workflow.BrowserWorkflowError("browser_workflow_session_unavailable")
        return identity

    def _session_valid(self, task: Any, state: dict[str, Any]) -> bool:
        session = self._sessions.get((task.user, task.task_id))
        try:
            return bool(self.enabled and not self.cancelled() and session and session ==
                        (state["session"], self.authorized(task.user, task.task_id), self._transport_identity(task)))
        except Exception:
            return False

    def _live(self, task: Any, *, mutation: bool = False) -> tuple[str, dict[str, Any]]:
        phase = self._check(task, mutation=mutation)
        state = workflow.load(task)
        if not self._session_valid(task, state):
            self._sessions.pop((task.user, task.task_id), None)
            self.private_data.clear_task(actor=task.user, task=task.task_id)
            raise workflow.BrowserWorkflowError("browser_workflow_session_unavailable")
        return phase, state

    def reacquire(self, task: Any) -> None:
        """Explicitly attach a separately verified task-owned session after loss.

        The transport must only attach a session already established by the
        authorized host. It must not create a page or repeat a prior effect.
        """
        with self._lock:
            self._check(task, mutation=False)
            prior = workflow.load(task)
            key = (task.user, task.task_id)
            if key in self._sessions or not callable(getattr(self.transport, "attach_owned_session", None)):
                raise workflow.BrowserWorkflowError("browser_workflow_reacquisition_unavailable")
            grant = self.authorized(task.user, task.task_id)
            try:
                attached = self.transport.attach_owned_session(task.user, task.task_id)
            except Exception:
                raise workflow.BrowserWorkflowError("browser_workflow_reacquisition_unavailable") from None
            self._check(task, mutation=False)
            if attached is not True or self.authorized(task.user, task.task_id) != grant:
                raise workflow.BrowserWorkflowError("browser_workflow_reacquisition_unavailable")
            identity = self._transport_identity(task)
            self.private_data.clear_task(actor=task.user, task=task.task_id)
            current = workflow.rotate_session(task)
            self.checkpoints.save_active_browser_checkpoint(task, expected_workflow=prior)
            self._sessions[key] = (current["session"], grant, identity)

    def supply(self, task: Any, *, actor_id: str, field: str, value: str, ttl: float = 300) -> None:
        """Accept private input only from an authenticated task owner."""
        with self._lock:
            if not actor_id or actor_id != task.user:
                raise workflow.BrowserWorkflowError("browser_workflow_input_denied")
            phase, state = self._live(task)
            if field not in workflow.FIELDS or not isinstance(value, str) or not value or len(value) > 4096:
                raise workflow.BrowserWorkflowError("browser_workflow_input_invalid")
            check = lambda: self._session_valid(task, state)
            ref = self.private_data.put(actor=task.user, task=task.task_id, grant_id=state["session"],
                                        payload=value, authorized=check, ttl=ttl)
            workflow.append(task, kind="supplied", field=field, phase=phase, ref=ref)
            self.checkpoints.save_active_browser_checkpoint(task, expected_workflow=state)

    def validate_private_request(self, task: Any, *, field: str) -> None:
        with self._lock:
            self._live(task)
            tactical = task.hierarchical_state
            phase = tactical["phase"]
            subgoal = next((s for s in phase.get("subgoals", []) if isinstance(s, dict)
                            and s.get("subgoal_id") == tactical.get("active_subgoal_id")), None)
            channel = task.metadata.get("channel") or {}
            if (field not in workflow.FIELDS or not isinstance(channel, dict) or channel.get("integration_id") != "desktop"
                    or not subgoal or "user_response" not in subgoal.get("allowed_side_effects", [])):
                raise workflow.BrowserWorkflowError("browser_workflow_question_denied")

    def record_requested_question(self, task: Any, *, question_id: str) -> None:
        """Retain a field category only for a recorded, owner-scoped private question.

        The private question/answer channel is supplied separately by the host;
        this method never reads or stores an answer.
        """
        with self._lock:
            phase, state = self._live(task)
            question = self.checkpoints.get_question(question_id)
            delivery = (question.metadata.get("delivery") or {}) if question is not None else {}
            field = delivery.get("browser_private_field") if isinstance(delivery, dict) else None
            if (question is None or question.task_id != task.task_id or question.originator_user_id != task.user
                    or question.respondent_user_id != task.user or question.status not in {"pending", "answered"}
                    or delivery.get("private_browser_value") is not True or field not in workflow.FIELDS):
                raise workflow.BrowserWorkflowError("browser_workflow_question_denied")
            if any(e["kind"] == "requested" and e["source"] == question_id for e in state["events"]):
                return
            workflow.append(task, kind="requested", field=field, phase=phase, source=question_id)
            self.checkpoints.save_active_browser_checkpoint(task, expected_workflow=state)

    def store_pending_answer(self, task: Any, *, actor_id: str, question_id: str,
                             value: str, ttl: float = 300) -> str:
        """Store an authenticated answer without writing it to the question store."""
        with self._lock:
            if (not self.enabled or self.cancelled() or not actor_id or actor_id != task.user
                    or task.status != "waiting_user" or task.metadata.get("pending_question_id") != question_id
                    or not isinstance(value, str) or not value or len(value) > 4096):
                raise workflow.BrowserWorkflowError("browser_workflow_input_denied")
            state = workflow.load(task)
            session = self._sessions.get((task.user, task.task_id))
            question = self.checkpoints.get_question(question_id)
            delivery = (question.metadata.get("delivery") or {}) if question is not None else {}
            field = delivery.get("browser_private_field") if isinstance(delivery, dict) else None
            try:
                unexpired = datetime.fromisoformat(question.expires_at) > datetime.now(timezone.utc)
            except (AttributeError, TypeError, ValueError):
                unexpired = False
            if (not session or not self._session_valid(task, state) or question is None or not unexpired
                    or question.status != "pending" or question.task_id != task.task_id
                    or question.originator_user_id != actor_id or question.respondent_user_id != actor_id
                    or delivery.get("private_browser_value") is not True or field not in workflow.FIELDS):
                raise workflow.BrowserWorkflowError("browser_workflow_input_denied")
            check = lambda: self._session_valid(task, state)
            return self.private_data.put(actor=actor_id, task=task.task_id, grant_id=state["session"],
                payload={"value": value, "field": field, "question_id": question_id}, authorized=check, ttl=ttl)

    def validate_pending_answer(self, actor: str, task_id: str, handle: str) -> bool:
        """Validate the live protected answer before the question store accepts it."""
        with self._lock:
            task = self.checkpoints.load_task_checkpoint(task_id)
            if task is None:
                return False
            try:
                state = workflow.load(task)
                session = self._sessions.get((actor, task_id))
                check = lambda: self._session_valid(task, state)
                if not self.enabled or task.user != actor or task.status != "waiting_user" or not session or not check():
                    return False
                payload = self.private_data.get(actor=actor, task=task_id, grant_id=state["session"],
                                                handle=handle, authorized=check)
                return (isinstance(payload, dict) and payload.get("question_id") == task.metadata.get("pending_question_id")
                        and payload.get("field") in workflow.FIELDS and isinstance(payload.get("value"), str))
            except Exception:
                return False

    def record_answered_question(self, task: Any, *, question_id: str) -> None:
        """Bind a resumed private answer to its owner and recorded question."""
        with self._lock:
            phase, state = self._live(task)
            question = self.checkpoints.get_question(question_id)
            interrupt = task.metadata.get("question_interrupt") or {}
            delivery = (question.metadata.get("delivery") or {}) if question is not None else {}
            field = delivery.get("browser_private_field") if isinstance(delivery, dict) else None
            answer = interrupt.get("answer") if isinstance(interrupt, dict) else None
            ref = answer.get("text") if isinstance(answer, dict) and set(answer) == {"text"} else None
            if (question is None or question.status != "answered" or question.task_id != task.task_id
                    or question.respondent_user_id != task.user or question.originator_user_id != task.user
                    or interrupt.get("question_id") != question_id or field not in workflow.FIELDS
                    or delivery.get("private_browser_value") is not True or not isinstance(ref, str)):
                raise workflow.BrowserWorkflowError("browser_workflow_question_denied")
            check = lambda: self._session_valid(task, state)
            payload = self.private_data.get(actor=task.user, task=task.task_id, grant_id=state["session"],
                                            handle=ref, authorized=check)
            if not isinstance(payload, dict) or payload.get("question_id") != question_id or payload.get("field") != field:
                raise workflow.BrowserWorkflowError("browser_workflow_question_denied")
            if not any(e["kind"] == "requested" and e["source"] == question_id for e in state["events"]):
                workflow.append(task, kind="requested", field=field, phase=phase, source=question_id)
            if not any(e["kind"] == "supplied" and e["source"] == question_id for e in state["events"]):
                workflow.append(task, kind="supplied", field=field, phase=phase, ref=ref, source=question_id)
            self.checkpoints.save_active_browser_checkpoint(task, expected_workflow=state)

    def observe(self, task: Any, *, result: bool = False) -> str:
        with self._lock:
            phase, state = self._live(task)
            try:
                page = self.transport.observe()
            except Exception:
                raise workflow.BrowserWorkflowError("browser_workflow_observation_failed") from None
            self._live(task)
            check = lambda: self._session_valid(task, state)
            ref = self.private_data.put(actor=task.user, task=task.task_id, grant_id=state["session"],
                                        payload=page, authorized=check, category="observation")
            workflow.append(task, kind="observed", field="result" if result else "form", phase=phase, ref=ref)
            self.checkpoints.save_active_browser_checkpoint(task, expected_workflow=state)
            return ref

    def fill(self, task: Any, *, field: str, target: str) -> None:
        with self._lock:
            phase, state = self._live(task, mutation=True)
            if field not in workflow.FIELDS or workflow.unresolved(state):
                raise workflow.BrowserWorkflowError("browser_workflow_reconciliation_required")
            supplied = next((e for e in reversed(state["events"]) if e["kind"] == "supplied" and e["field"] == field and e["session"] == state["session"]), None)
            if supplied is None:
                raise workflow.BrowserWorkflowError("browser_workflow_input_unavailable")
            observed = next((e for e in reversed(state["events"]) if e["kind"] == "observed" and e["field"] == "form" and e["session"] == state["session"]), None)
            if observed is None or not callable(getattr(self.transport, "fill_if_unchanged", None)):
                raise workflow.BrowserWorkflowError("browser_workflow_observation_required")
            check = lambda: self._session_valid(task, state)
            value = self.private_data.get(actor=task.user, task=task.task_id, grant_id=state["session"],
                                          handle=supplied["ref"], authorized=check)
            if isinstance(value, dict):
                if value.get("field") != field or value.get("question_id") != supplied["source"]:
                    raise workflow.BrowserWorkflowError("browser_workflow_input_unavailable")
                value = value.get("value")
            if not isinstance(value, str) or not value:
                raise workflow.BrowserWorkflowError("browser_workflow_input_unavailable")
            self._reject_verified_fill_replay(task, state, field=field, target=target, value=value, authorized=check)
            page = self.private_data.get(actor=task.user, task=task.task_id, grant_id=state["session"],
                                         handle=observed["ref"], authorized=check)
            if not isinstance(page, dict) or not isinstance(page.get("targets"), list) or target not in page["targets"]:
                raise workflow.BrowserWorkflowError("browser_workflow_target_unavailable")
            binding = self.private_data.put(actor=task.user, task=task.task_id, grant_id=state["session"],
                payload={"target": target, "value_ref": supplied["ref"], "field": field},
                authorized=check, category="binding")
            attempt = workflow.append(task, kind="attempted", field="fill", phase=phase, ref=binding)
            self.checkpoints.save_active_browser_checkpoint(task, expected_workflow=state)
            self._live(task, mutation=True)
            try:
                # The transport must fence the fill against the exact observation
                # in the same serialized browser operation. A transport lacking
                # this contract is not eligible to execute workflow mutations.
                if self.transport.fill_if_unchanged(target, value, page) is not True:
                    raise workflow.BrowserWorkflowError("browser_workflow_page_changed")
            except Exception:
                raise workflow.BrowserWorkflowError("browser_workflow_effect_unknown") from None
            self._live(task, mutation=True)
            before_return = workflow.load(task)
            workflow.append(task, kind="returned", field="fill", phase=phase, related=attempt)
            self.checkpoints.save_active_browser_checkpoint(task, expected_workflow=before_return)

    def _reject_verified_fill_replay(self, task: Any, state: dict[str, Any], *, field: str,
                                     target: str, value: str, authorized: Callable[[], bool]) -> None:
        """Compare identities in protected storage; never infer equality from redaction."""
        events = state["events"]
        attempts = {e["id"]: e for e in events if e["kind"] == "attempted" and e["field"] == "fill"}
        for event in events:
            if event["kind"] != "verified" or event["related"] not in attempts:
                continue
            attempt = attempts[event["related"]]
            if attempt["session"] != state["session"]:
                raise workflow.BrowserWorkflowError("browser_workflow_reconciliation_required")
            try:
                binding = self.private_data.get(actor=task.user, task=task.task_id, grant_id=state["session"],
                                                handle=attempt["ref"], authorized=authorized)
                prior = self.private_data.get(actor=task.user, task=task.task_id, grant_id=state["session"],
                                              handle=binding["value_ref"], authorized=authorized)
                prior_value = prior.get("value") if isinstance(prior, dict) else prior
                if (not isinstance(binding, dict) or binding.get("field") not in workflow.FIELDS
                        or not isinstance(binding.get("target"), str) or not isinstance(prior_value, str)):
                    raise ValueError("invalid_binding")
            except Exception:
                raise workflow.BrowserWorkflowError("browser_workflow_reconciliation_required") from None
            if (binding["field"], binding["target"], prior_value) == (field, target, value):
                raise workflow.BrowserWorkflowError("browser_workflow_verified_fill_replay")

    def submit(self, task: Any, *, target: str) -> None:
        """Write ahead of a submission and retain uncertainty until verified."""
        with self._lock:
            phase, state = self._live(task, mutation=True)
            if workflow.unresolved(state):
                raise workflow.BrowserWorkflowError("browser_workflow_reconciliation_required")
            if any(e["kind"] == "attempted" and e["field"] == "submit" for e in state["events"]):
                raise workflow.BrowserWorkflowError("browser_workflow_submission_already_attempted")
            observed = next((e for e in reversed(state["events"]) if e["kind"] == "observed"
                             and e["field"] == "form" and e["session"] == state["session"]), None)
            if observed is None or not callable(getattr(self.transport, "submit_if_unchanged", None)):
                raise workflow.BrowserWorkflowError("browser_workflow_observation_required")
            check = lambda: self._session_valid(task, state)
            page = self.private_data.get(actor=task.user, task=task.task_id, grant_id=state["session"],
                                         handle=observed["ref"], authorized=check)
            if not isinstance(page, dict) or not isinstance(page.get("targets"), list) or target not in page["targets"]:
                raise workflow.BrowserWorkflowError("browser_workflow_target_unavailable")
            binding = self.private_data.put(actor=task.user, task=task.task_id, grant_id=state["session"],
                payload={"target": target, "before_ref": observed["ref"]}, authorized=check, category="binding")
            attempt = workflow.append(task, kind="attempted", field="submit", phase=phase, ref=binding)
            self.checkpoints.save_active_browser_checkpoint(task, expected_workflow=state)
            self._live(task, mutation=True)
            try:
                if self.transport.submit_if_unchanged(target, page) is not True:
                    raise workflow.BrowserWorkflowError("browser_workflow_page_changed")
            except Exception:
                raise workflow.BrowserWorkflowError("browser_workflow_effect_unknown") from None
            self._live(task, mutation=True)
            prior = workflow.load(task)
            workflow.append(task, kind="returned", field="submit", phase=phase, related=attempt)
            self.checkpoints.save_active_browser_checkpoint(task, expected_workflow=prior)

    def verify_submission(self, task: Any) -> None:
        """Record only submission evidence tied to a fresh protected result."""
        with self._lock:
            phase, state = self._live(task)
            events = state["events"]
            resolved = {e["related"] for e in events if e["kind"] in {"verified", "reconciled_no_effect"}}
            pending = [(i, e) for i, e in enumerate(events)
                       if e["kind"] == "attempted" and e["field"] == "submit" and e["id"] not in resolved]
            if len(pending) != 1 or not callable(getattr(self.transport, "verify_submission", None)):
                raise workflow.BrowserWorkflowError("browser_workflow_reconciliation_required")
            index, attempt = pending[0]
            if attempt["session"] != state["session"]:
                raise workflow.BrowserWorkflowError("browser_workflow_reconciliation_required")
            result = next((e for e in reversed(events[index + 1:]) if e["kind"] == "observed"
                           and e["field"] == "result" and e["session"] == state["session"]), None)
            if result is None:
                raise workflow.BrowserWorkflowError("browser_workflow_observation_required")
            check = lambda: self._session_valid(task, state)
            binding = self.private_data.get(actor=task.user, task=task.task_id, grant_id=state["session"],
                                            handle=attempt["ref"], authorized=check)
            before = self.private_data.get(actor=task.user, task=task.task_id, grant_id=state["session"],
                                           handle=binding["before_ref"], authorized=check)
            after = self.private_data.get(actor=task.user, task=task.task_id, grant_id=state["session"],
                                          handle=result["ref"], authorized=check)
            try:
                verified = self.transport.verify_submission(binding["target"], before, after)
            except Exception:
                raise workflow.BrowserWorkflowError("browser_workflow_reconciliation_failed") from None
            self._live(task)
            if verified is not True:
                raise workflow.BrowserWorkflowError("browser_workflow_reconciliation_required")
            workflow.append(task, kind="verified", field="submit", phase=phase, ref=result["ref"], related=attempt["id"])
            self.checkpoints.save_active_browser_checkpoint(task, expected_workflow=state)

    def reconcile_fill(self, task: Any) -> None:
        """Verify a fill from a later observation; absence never proves no effect."""
        with self._lock:
            phase, state = self._live(task)
            events = state["events"]
            resolved = {e["related"] for e in events if e["kind"] in {"verified", "reconciled_no_effect"}}
            pending = [(i, e) for i, e in enumerate(events)
                       if e["kind"] == "attempted" and e["field"] == "fill" and e["id"] not in resolved]
            if len(pending) != 1 or not callable(getattr(self.transport, "verify_fill", None)):
                raise workflow.BrowserWorkflowError("browser_workflow_reconciliation_required")
            index, attempt = pending[0]
            if attempt["session"] != state["session"]:
                raise workflow.BrowserWorkflowError("browser_workflow_reconciliation_required")
            observation = next((e for e in reversed(events[index + 1:])
                                if e["kind"] == "observed" and e["field"] == "form"), None)
            if observation is None or attempt["ref"] is None:
                raise workflow.BrowserWorkflowError("browser_workflow_observation_required")
            check = lambda: self._session_valid(task, state)
            binding = self.private_data.get(actor=task.user, task=task.task_id, grant_id=state["session"],
                                            handle=attempt["ref"], authorized=check)
            value = self.private_data.get(actor=task.user, task=task.task_id, grant_id=state["session"],
                                          handle=binding["value_ref"], authorized=check)
            if isinstance(value, dict):
                if value.get("field") != binding["field"]:
                    raise workflow.BrowserWorkflowError("browser_workflow_input_unavailable")
                value = value.get("value")
            if not isinstance(value, str) or not value:
                raise workflow.BrowserWorkflowError("browser_workflow_input_unavailable")
            page = self.private_data.get(actor=task.user, task=task.task_id, grant_id=state["session"],
                                         handle=observation["ref"], authorized=check)
            try:
                verified = self.transport.verify_fill(binding["target"], value, page)
            except Exception:
                raise workflow.BrowserWorkflowError("browser_workflow_reconciliation_failed") from None
            self._live(task)
            if verified is not True:
                raise workflow.BrowserWorkflowError("browser_workflow_reconciliation_required")
            workflow.append(task, kind="verified", field=binding["field"], phase=phase,
                            ref=observation["ref"], related=attempt["id"])
            self.checkpoints.save_active_browser_checkpoint(task, expected_workflow=state)

    def disconnect(self, task: Any) -> None:
        with self._lock:
            self._sessions.pop((task.user, task.task_id), None)
            self.private_data.clear_task(actor=task.user, task=task.task_id)

    def projection(self, task: Any, *, limit: int = 64) -> dict[str, Any]:
        """Return historical facts and live handle availability, never their payloads."""
        def available(state: dict[str, Any], event: dict[str, Any]) -> bool:
            session = self._sessions.get((task.user, task.task_id))
            if event["session"] != state["session"] or not session or not self._session_valid(task, state):
                return False
            check = lambda: self._session_valid(task, state)
            self.private_data.get(actor=task.user, task=task.task_id, grant_id=state["session"],
                                  handle=event["ref"], authorized=check)
            return True
        with self._lock:
            state = workflow.load(task)
            try:
                session_live = self._session_valid(task, state)
            except Exception:
                session_live = False
            return workflow.projection(task, available=available, session_available=session_live, limit=limit)
