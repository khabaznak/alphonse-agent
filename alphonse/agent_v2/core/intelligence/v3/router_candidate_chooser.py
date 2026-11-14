"""Candidate chooser using the configured inference router.

This adapter does not register an engine or authorize tools. Its caller must
authenticate the task owner, select allowed read-only tools, and enforce access,
argument validation, cancellation and durable publication independently.
Visible byte bounds are not provider token limits.
"""
from __future__ import annotations

from copy import deepcopy
import json
from typing import Callable

from alphonse.agent_v2.core.core import ToolDescriptor, ToolKind
from alphonse.agent_v2.core.inference.models import InferenceResult
from alphonse.agent_v2.core.inference.openai_codex import _json_envelope
from alphonse.agent_v2.core.inference.router import InferenceRouter
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from .candidate_decision_policy import action_schema, make_request, normalize_action, validate_source_progress


class RouterCandidateChooser:
    """Choose a candidate action against an explicitly supplied tool catalog."""

    def __init__(self, *, router: InferenceRouter, task: TaskState, tools: tuple[ToolDescriptor, ...],
                 max_input_tokens: int, cancellation_checker: Callable[[], bool] | None = None,
                 observation_progress: Callable[[], dict[str, list[str]]] | None = None):
        if not isinstance(router, InferenceRouter) or type(max_input_tokens) is not int or max_input_tokens <= 0:
            raise ValueError('candidate_router_configuration_invalid')
        if (not isinstance(task, TaskState) or not isinstance(task.task_id, str) or not task.task_id.strip()
                or not isinstance(task.user, str) or not task.user.strip()
                or not isinstance(task.project_id, str)):
            raise ValueError('candidate_router_task_identity_invalid')
        if cancellation_checker is not None and not callable(cancellation_checker):
            raise ValueError('candidate_router_configuration_invalid')
        if observation_progress is not None and not callable(observation_progress):
            raise ValueError('candidate_router_configuration_invalid')
        if not isinstance(tools, tuple):
            raise ValueError('candidate_router_tools_invalid')
        catalog = []
        seen = set()
        for tool in tools:
            if (not isinstance(tool, ToolDescriptor) or tool.kind is not ToolKind.NATIVE
                    or tool.read_only is not True or not isinstance(tool.tool_id, str)
                    or not tool.tool_id.strip() or tool.tool_id in seen
                    or not isinstance(tool.description, str) or not isinstance(tool.argument_schema, dict)):
                raise ValueError('candidate_router_tools_invalid')
            seen.add(tool.tool_id)
            catalog.append({'tool_id': tool.tool_id, 'description': tool.description,
                            'argument_schema': deepcopy(tool.argument_schema)})
        try:
            self._catalog_json = json.dumps(catalog, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError('candidate_router_tools_invalid') from exc
        self._catalog = catalog
        self._tool_ids = frozenset(seen)
        self.router = router
        self._task = task
        self._identity = (task.task_id, task.user, task.project_id)
        self.max_input_tokens = max_input_tokens
        self.cancellation_checker = cancellation_checker
        # Optional host callback must provide only currently authorized labels
        # for recorded observations and discoverable unread sources. The chooser
        # validates structure, not the host's authority or data provenance.
        self.observation_progress = observation_progress

    def _cancelled(self) -> bool:
        if self.cancellation_checker is None:
            return False
        try:
            return self.cancellation_checker() is not False
        except Exception:
            return True

    def __call__(self, projection: str, max_output_tokens: int) -> dict:
        if not isinstance(projection, str) or type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise ValueError('candidate_router_request_invalid')
        if self._cancelled():
            raise ValueError('candidate_router_cancelled')
        if (self._task.task_id, self._task.user, self._task.project_id) != self._identity:
            raise ValueError('candidate_router_task_identity_changed')
        try:
            progress = (validate_source_progress(self.observation_progress())
                        if self.observation_progress is not None else None)
            request = make_request(projection=projection, catalog=self._catalog,
                output_schema=action_schema(), cancellation_checker=self.cancellation_checker,
                task_id=self._identity[0], user=self._identity[1], project_id=self._identity[2],
                source_progress=progress)
        except Exception:
            raise ValueError('candidate_router_source_progress_unavailable') from None
        if (self._task.task_id, self._task.user, self._task.project_id) != self._identity:
            raise ValueError('candidate_router_task_identity_changed')
        if self._cancelled():
            raise ValueError('candidate_router_cancelled')
        # This matches the existing Codex JSON transport, but cannot attest
        # framing of another provider or hidden provider tokens.
        if len(_json_envelope(request).encode('utf-8')) > self.max_input_tokens:
            raise ValueError('candidate_router_input_exceeds_reservation')
        failed = False
        try:
            result = self.router.generate_json(request)
        except Exception:
            failed = True
        if failed:
            raise ValueError('candidate_router_unavailable')
        if self._cancelled():
            raise ValueError('candidate_router_cancelled')
        if (self._task.task_id, self._task.user, self._task.project_id) != self._identity:
            raise ValueError('candidate_router_task_identity_changed')
        if not isinstance(result, InferenceResult) or not isinstance(result.json_value, dict):
            raise ValueError('candidate_router_result_invalid')
        try:
            rendered = json.dumps(result.json_value, ensure_ascii=False, allow_nan=False)
        except (ValueError, TypeError):
            raise ValueError('candidate_router_result_invalid') from None
        raw = result.raw_response if isinstance(result.raw_response, str) else ''
        if max(len(rendered.encode('utf-8')), len(raw.encode('utf-8'))) > max_output_tokens:
            raise ValueError('candidate_router_output_exceeds_reservation')
        decision = normalize_action(result.json_value, self._tool_ids)
        usage = result.usage if isinstance(result.usage, dict) else {}
        decision['usage'] = {key: usage.get(key) if type(usage.get(key)) is int and usage[key] >= 0 else None
                             for key in ('input_tokens', 'output_tokens')}
        return decision
