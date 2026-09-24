"""TypeSafe Jev configuration, HTTP client, and bounded criterion decisions."""

from __future__ import annotations

import json
import hashlib
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from time import monotonic
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from alphonse.agent_v2.database import connect_database, default_database_path

DEFAULT_SYSTEM_ONE_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_SYSTEM_ONE_MODEL = "jev-latest"
_JEV_TOOL_REGISTRY_PATH = Path(__file__).resolve().parent / "config" / "jev_tool_registry.json"
_TACTICAL_RETRY_FUSE_QUESTIONS = (
    {
        "question_id": "retry_fuse_transient_failure",
        "type": "noul",
        "instructions": "Does the observed tool failure appear transient rather than caused by invalid credentials, missing permissions, invalid input, or another persistent configuration problem?",
        "criteria": {
            "true": "Evidence points to a temporary condition such as a timeout, connection interruption, or temporary service outage.",
            "false": "Evidence points to credentials, permissions, invalid input, unavailable configuration, or another condition that will persist without a change.",
        },
    },
    {
        "question_id": "retry_fuse_same_call_likely_to_work",
        "type": "noul",
        "instructions": "After a short backoff, is the same tool call with exactly the same arguments reasonably likely to succeed without new information or configuration?",
        "criteria": {
            "true": "A temporary failure is plausible and another identical attempt could succeed without changing inputs.",
            "false": "The call needs changed arguments, credentials, permissions, configuration, or user input to succeed.",
        },
    },
    {
        "question_id": "retry_fuse_repeat_is_safe",
        "type": "noul",
        "instructions": "Is repeating this exact tool call safe, without a meaningful risk of duplicating or compounding side effects?",
        "criteria": {
            "true": "The operation is read-only/idempotent, or the execution evidence explicitly establishes safe retry semantics.",
            "false": "The operation may have partially succeeded, may duplicate side effects, or its retry safety is unknown.",
        },
    },
)


class SystemOneUnavailableError(RuntimeError):
    """Jev is required for V3 decisions but the service cannot answer."""

    def __init__(self, detail: str = "") -> None:
        suffix = f":{detail}" if detail else ""
        super().__init__(f"system_one_unavailable{suffix}")


@dataclass(frozen=True)
class SystemOneSettings:
    enabled: bool = False
    api_url: str = DEFAULT_SYSTEM_ONE_URL
    model: str = DEFAULT_SYSTEM_ONE_MODEL
    api_key: str = ""
    yes_threshold: float = 0.80
    no_threshold: float = 0.20
    route_confidence_threshold: float = 0.70
    validated_at: str = ""
    validation_error: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "api_url": self.api_url,
            "model": self.model,
            "has_api_key": bool(self.api_key),
            "yes_threshold": self.yes_threshold,
            "no_threshold": self.no_threshold,
            "route_confidence_threshold": self.route_confidence_threshold,
            "validated_at": self.validated_at,
            "validation_error": self.validation_error,
            "updated_at": self.updated_at,
        }


class SQLiteSystemOneSettingsStore:
    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._memory_connection: sqlite3.Connection | None = None
        if self.db_path == ":memory:":
            self._memory_connection = sqlite3.connect(":memory:", check_same_thread=False)
            self._memory_connection.row_factory = sqlite3.Row
        self._ensure_schema()

    @classmethod
    def default(cls) -> "SQLiteSystemOneSettingsStore":
        return cls(default_database_path())

    def get(self) -> SystemOneSettings:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM v2_system_one_settings WHERE settings_id=1").fetchone()
        if row is None:
            return SystemOneSettings()
        return SystemOneSettings(
            enabled=bool(row["enabled"]), api_url=str(row["api_url"]), model=str(row["model"]),
            api_key=str(row["api_key"]), yes_threshold=float(row["yes_threshold"]),
            no_threshold=float(row["no_threshold"]),
            route_confidence_threshold=float(row["route_confidence_threshold"]),
            validated_at=str(row["validated_at"]), validation_error=str(row["validation_error"]),
            updated_at=str(row["updated_at"]),
        )

    def save(self, settings: SystemOneSettings) -> SystemOneSettings:
        normalized = _normalize_settings(settings)
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO v2_system_one_settings
                (settings_id, enabled, api_url, model, api_key, yes_threshold, no_threshold,
                 route_confidence_threshold, validated_at, validation_error, updated_at)
                VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (int(normalized.enabled), normalized.api_url, normalized.model, normalized.api_key,
                 normalized.yes_threshold, normalized.no_threshold, normalized.route_confidence_threshold,
                 normalized.validated_at, normalized.validation_error, _now()),
            )
        return self.get()

    def _connect(self):
        if self._memory_connection is not None:
            return _ConnectionProxy(self._memory_connection)
        path = Path(self.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return connect_database(path)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS v2_system_one_settings (
              settings_id INTEGER PRIMARY KEY CHECK (settings_id=1),
              enabled INTEGER NOT NULL DEFAULT 0,
              api_url TEXT NOT NULL,
              model TEXT NOT NULL,
              api_key TEXT NOT NULL DEFAULT '',
              yes_threshold REAL NOT NULL DEFAULT 0.8,
              no_threshold REAL NOT NULL DEFAULT 0.2,
              route_confidence_threshold REAL NOT NULL DEFAULT 0.7,
              validated_at TEXT NOT NULL DEFAULT '',
              validation_error TEXT NOT NULL DEFAULT '',
              updated_at TEXT NOT NULL
            ) STRICT;
            """)


Transport = Callable[[str, str, dict[str, Any], float], dict[str, Any]]


class TypeSafeSystemOneClient:
    def __init__(self, *, api_url: str, api_key: str, model: str, timeout_seconds: float = 15.0, transport: Transport | None = None) -> None:
        self.api_url = _normalize_url(api_url)
        self.api_key = str(api_key or "").strip()
        self.model = str(model or "").strip() or DEFAULT_SYSTEM_ONE_MODEL
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.transport = transport or _http_transport

    def evaluate(self, *, state: Any, questions: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise ValueError("system_one_api_key_required")
        if not questions:
            raise ValueError("system_one_questions_required")
        response = self.transport(
            self.api_url, self.api_key,
            {"state": state, "model": self.model, "questions": questions},
            self.timeout_seconds,
        )
        answers = response.get("answers") if isinstance(response, dict) else None
        if not isinstance(answers, dict):
            raise ValueError("system_one_response_invalid")
        return response

    def validate(self) -> dict[str, Any]:
        response = self.evaluate(
            state={"connection_test": "Alphonse System One settings validation"},
            questions={"ready": {"type": "noul", "instructions": "Is this a connection-test state?"}},
        )
        answer = response["answers"].get("ready")
        if not isinstance(answer, dict) or answer.get("type") != "noul" or not isinstance(answer.get("noul"), (int, float)):
            raise ValueError("system_one_validation_response_invalid")
        return response


@dataclass(frozen=True)
class SystemOneReviewResult:
    updates: tuple[dict[str, Any], ...]
    ambiguous_criterion_ids: tuple[str, ...]
    recommended_route: str = ""
    route_confidence: float = 0.0
    route_confident: bool = False
    model: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    duration_ms: int = 0

    def to_metadata(self) -> dict[str, Any]:
        return {
            "updates": [dict(item) for item in self.updates],
            "ambiguous_criterion_ids": list(self.ambiguous_criterion_ids),
            "recommended_route": self.recommended_route,
            "route_confidence": self.route_confidence,
            "route_confident": self.route_confident,
            "model": self.model,
            "usage": dict(self.usage),
            "duration_ms": self.duration_ms,
        }


@dataclass(frozen=True)
class SystemOneToolRegistrySelection:
    selected_tool_ids: tuple[str, ...] = ()
    rejected_tool_ids: tuple[str, ...] = ()
    ambiguous_tool_ids: tuple[str, ...] = ()
    probabilities: dict[str, float] = field(default_factory=dict)
    model: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    duration_ms: int = 0

    def to_metadata(self) -> dict[str, Any]:
        return {
            "selected_tool_ids": list(self.selected_tool_ids),
            "rejected_tool_ids": list(self.rejected_tool_ids),
            "ambiguous_tool_ids": list(self.ambiguous_tool_ids),
            "probabilities": dict(self.probabilities),
            "model": self.model,
            "usage": dict(self.usage),
            "duration_ms": self.duration_ms,
        }


@dataclass(frozen=True)
class SystemOneTacticalReview:
    complete: bool
    confidence: float
    confident: bool
    model: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    duration_ms: int = 0
    answers: dict[str, float] = field(default_factory=dict)
    retry_approved: bool = False

    def to_metadata(self) -> dict[str, Any]:
        return {
            "complete": self.complete,
            "confidence": self.confidence,
            "confident": self.confident,
            "model": self.model,
            "usage": dict(self.usage),
            "duration_ms": self.duration_ms,
            "answers": dict(self.answers),
            "retry_approved": self.retry_approved,
        }


@dataclass(frozen=True)
class SystemOneActRecommendation:
    action: str
    confidence: float
    confident: bool
    rationale: str
    answers: dict[str, float] = field(default_factory=dict)
    yes_threshold: float = 0.8
    model: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    duration_ms: int = 0

    def to_metadata(self) -> dict[str, Any]:
        return {
            "action": self.action, "confidence": self.confidence, "confident": self.confident,
            "rationale": self.rationale, "model": self.model, "usage": dict(self.usage),
            "duration_ms": self.duration_ms, "answers": dict(self.answers), "yes_threshold": self.yes_threshold,
        }


@dataclass(frozen=True)
class SystemOneAdmissionDecision:
    """One-time judgment for routing a new human message into V3."""

    requires_task: bool
    confidence: float
    confident: bool
    model: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    duration_ms: int = 0

    def to_metadata(self) -> dict[str, Any]:
        return {
            "requires_task": self.requires_task,
            "confidence": self.confidence,
            "confident": self.confident,
            "model": self.model,
            "usage": dict(self.usage),
            "duration_ms": self.duration_ms,
        }


@dataclass(frozen=True)
class _StaticJevToolRegistry:
    signature: tuple[str, ...]
    questions: dict[str, Any]
    tool_ids_by_question: dict[str, str]


class JevCriterionDecisionProvider:
    def __init__(self, settings: SystemOneSettings, *, transport: Transport | None = None) -> None:
        self.settings = _normalize_settings(settings)
        self.client = TypeSafeSystemOneClient(
            api_url=self.settings.api_url, api_key=self.settings.api_key,
            model=self.settings.model, transport=transport,
        )
        self._tool_registry: _StaticJevToolRegistry | None = None

    def classify_task_admission(self, *, message: str) -> SystemOneAdmissionDecision:
        """Decide whether a new human message needs work beyond one direct reply."""
        state = {"user_message": str(message or "")[:4000]}
        questions = {
            "requires_task": {
                "type": "noul",
                "instructions": (
                    "Does satisfying this user message require work beyond one immediate conversational text reply?"
                ),
                "criteria": {
                    "true": (
                        "Satisfying it requires planning, extended analysis, retrieval, tools, verification, "
                        "a mutation, scheduling, communication, an external action, or continued work."
                    ),
                    "false": (
                        "One immediate conversational text reply fully satisfies it without planning, retrieval, "
                        "tools, verification, mutation, scheduling, communication, or continued work."
                    ),
                },
            }
        }
        started = monotonic()
        response = self.client.evaluate(state=state, questions=questions)
        duration_ms = max(0, round((monotonic() - started) * 1000))
        answer = response["answers"].get("requires_task")
        if not isinstance(answer, dict) or answer.get("type") != "noul" or not isinstance(answer.get("noul"), (int, float)):
            raise ValueError("system_one_admission_answer_invalid")
        probability = max(0.0, min(1.0, float(answer["noul"])))
        return SystemOneAdmissionDecision(
            requires_task=probability >= self.settings.yes_threshold,
            confidence=probability,
            confident=probability >= self.settings.yes_threshold or probability <= self.settings.no_threshold,
            model=str(response.get("model") or self.settings.model),
            usage=dict(response.get("usage") or {}),
            duration_ms=duration_ms,
        )

    def evaluate(self, *, contract: dict[str, Any], phase: dict[str, Any], evidence: dict[str, Any]) -> SystemOneReviewResult:
        criteria = [
            item for item in contract.get("criteria") or []
            if isinstance(item, dict) and item.get("superseded") is not True
            and item.get("required", True) and item.get("verification") != "system"
            and item.get("status") != "satisfied"
        ]
        entries = _bounded_evidence(evidence.get("entries") if isinstance(evidence, dict) else [])
        if not criteria:
            return SystemOneReviewResult(updates=(), ambiguous_criterion_ids=(), model=self.settings.model)
        state = {
            "acceptance_criteria": {str(item.get("id")): str(item.get("statement")) for item in criteria},
            "phase": {"phase_id": phase.get("phase_id"), "objective": phase.get("objective")},
            "verified_evidence": {item["evidence_ref"]: item["summary"] for item in entries},
        }
        questions: dict[str, Any] = {}
        pair_keys: dict[str, tuple[str, str]] = {}
        for criterion_index, criterion in enumerate(criteria):
            criterion_id = str(criterion.get("id") or "")
            for evidence_index, entry in enumerate(entries):
                key = f"supports_{criterion_index}_{evidence_index}"
                pair_keys[key] = (criterion_id, entry["evidence_ref"])
                questions[key] = {
                    "type": "noul",
                    "instructions": f"Does verified evidence {entry['evidence_ref']} directly demonstrate acceptance criterion {criterion_id}?",
                    "criteria": {
                        "true": "The evidence directly observes or verifies the criterion's required outcome.",
                        "false": "The evidence is unrelated, merely planned, failed, ambiguous, or does not verify the outcome.",
                    },
                }
        started = monotonic()
        response = self.client.evaluate(state=state, questions=questions)
        duration_ms = max(0, round((monotonic() - started) * 1000))
        answers = response["answers"]
        support: dict[str, list[tuple[str, float]]] = {str(item.get("id")): [] for item in criteria}
        for key, (criterion_id, evidence_ref) in pair_keys.items():
            answer = answers.get(key)
            if not isinstance(answer, dict) or answer.get("type") != "noul" or not isinstance(answer.get("noul"), (int, float)):
                raise ValueError(f"system_one_answer_invalid:{key}")
            support[criterion_id].append((evidence_ref, float(answer["noul"])))
        updates: list[dict[str, Any]] = []
        ambiguous: list[str] = []
        for criterion in criteria:
            criterion_id = str(criterion.get("id") or "")
            refs = [ref for ref, probability in support.get(criterion_id, []) if probability >= self.settings.yes_threshold]
            probabilities = [probability for _, probability in support.get(criterion_id, [])]
            if refs:
                updates.append({"criterion_id": criterion_id, "status": "satisfied", "evidence_refs": refs, "reason": "System One evidence decision"})
            elif probabilities and all(probability <= self.settings.no_threshold for probability in probabilities):
                updates.append({"criterion_id": criterion_id, "status": "pending", "evidence_refs": [], "reason": "No supporting evidence identified"})
            elif probabilities:
                ambiguous.append(criterion_id)
            else:
                updates.append({"criterion_id": criterion_id, "status": "pending", "evidence_refs": [], "reason": "No verified evidence available"})
        return SystemOneReviewResult(
            updates=tuple(updates), ambiguous_criterion_ids=tuple(ambiguous),
            model=str(response.get("model") or self.settings.model),
            usage=dict(response.get("usage") or {}), duration_ms=duration_ms,
        )

    def triage_plan_messages(self, *, task: dict[str, Any], candidates: list[dict[str, Any]]) -> tuple[str, ...]:
        """Return queued message IDs Jev judges relevant to the active task."""
        if not candidates:
            return ()
        state = {
            "active_task": {
                "task_id": task.get("task_id"),
                "goal": task.get("goal"),
                "acceptance_contract": task.get("acceptance_contract"),
                "strategic_plan": task.get("strategic_plan"),
            },
            "queued_messages": [
                {"message_id": str(item["message_id"]), "sender": item.get("sender"), "text": item.get("text"),
                 "correlated_question_id": item.get("question_id")}
                for item in candidates
            ],
        }
        questions: dict[str, Any] = {}
        key_to_id: dict[str, str] = {}
        for index, item in enumerate(candidates):
            key = f"message_class_{index}"
            key_to_id[key] = str(item["message_id"])
            questions[key] = {
                "type": "choice",
                "instructions": "Classify this queued message only by its relevance to the active task.",
                "criteria": {
                    "steering": "It changes, clarifies, or adds a requirement to the active task.",
                    "question_answer": "It answers an open question that blocks or informs the active task.",
                    "relevant_context": "It provides information or a dependency needed to complete the active task.",
                    "independent_task": "It requests separate work that belongs to a different task.",
                    "unrelated": "It is unrelated to the active task or is not actionable context for it.",
                },
            }
            questions[f"relevant_{index}"] = {
                "type": "noul",
                "instructions": f"Is queued message {item['message_id']} relevant to the active task?",
                "criteria": {
                    "true": "The message should be incorporated before planning or continuing this task.",
                    "false": "The message is unrelated, independent work, or should remain queued for another task.",
                },
            }
        response = self.client.evaluate(state=state, questions=questions)
        selected: list[str] = []
        answers = response["answers"]
        for index, (key, message_id) in enumerate(key_to_id.items()):
            choice = answers.get(key)
            fuse = answers.get(f"relevant_{index}")
            if not isinstance(choice, dict) or choice.get("type") != "choice":
                raise ValueError(f"system_one_message_classification_invalid:{key}")
            if not isinstance(fuse, dict) or fuse.get("type") != "noul" or not isinstance(fuse.get("noul"), (int, float)):
                raise ValueError(f"system_one_message_relevance_invalid:{index}")
            label = str(choice.get("choice") or "")
            probability = float((choice.get("probabilities") or {}).get(label) or 0.0)
            if label in {"steering", "question_answer", "relevant_context"} and probability >= self.settings.route_confidence_threshold and float(fuse["noul"]) >= self.settings.yes_threshold:
                selected.append(message_id)
        return tuple(selected)

    def recommend_act(self, *, state: dict[str, Any]) -> SystemOneActRecommendation:
        """Recommend the next mission-level action from Check's verdict and durable evidence."""
        actions = {
            "complete": "All required acceptance criteria are verified and the task can end successfully.",
            "continue": "More authorized work under the current strategy is likely to make useful progress.",
            "replan": "The current strategic approach should change before any further execution.",
            "ask_user": "A specific answer or steering from the user could materially unblock or redirect the mission.",
            "fail_explain": "The mission is no longer worth pursuing, but the user should receive a final evidence-based explanation.",
            "fail": "The mission must stop immediately and no further user-facing closure cycle is appropriate.",
        }
        questions = {
            "recommended_action": {
                "type": "choice",
                "instructions": "Recommend the next mission-level action, considering Check's verdict and all supplied evidence and constraints.",
                "criteria": actions,
            },
            "continuation_is_worthwhile": {
                "type": "noul",
                "instructions": "Would another bounded execution cycle likely create meaningful mission progress under the current strategy?",
                "criteria": {"true": "There is a plausible, authorized next step.", "false": "The same approach is exhausted, blocked, or unlikely to change the outcome."},
            },
            "user_input_can_unblock": {
                "type": "noul",
                "instructions": "Could a concrete user answer or steering materially change the mission's prospects?",
                "criteria": {"true": "A user decision or missing information is a plausible path forward.", "false": "User input would not change the feasibility or safety of the mission."},
            },
            "closure_explanation_is_warranted": {
                "type": "noul",
                "instructions": "If the mission ends unsuccessfully, should Alphonse spend one bounded final cycle explaining why to the user?",
                "criteria": {"true": "A concise explanation of the failure is useful and safe to deliver.", "false": "No additional user-facing closure cycle is appropriate."},
            },
        }
        started = monotonic()
        response = self.client.evaluate(state=state, questions=questions)
        duration_ms = max(0, round((monotonic() - started) * 1000))
        answer = response["answers"].get("recommended_action")
        if not isinstance(answer, dict) or answer.get("type") != "choice":
            raise ValueError("system_one_act_recommendation_invalid")
        action = str(answer.get("choice") or "")
        if action not in actions:
            raise ValueError("system_one_act_action_invalid")
        probabilities = answer.get("probabilities") if isinstance(answer.get("probabilities"), dict) else {}
        confidence = max(0.0, min(1.0, float(probabilities.get(action) or 0.0)))
        fuse_answers: dict[str, float] = {}
        for question_id in ("continuation_is_worthwhile", "user_input_can_unblock", "closure_explanation_is_warranted"):
            fuse = response["answers"].get(question_id)
            if not isinstance(fuse, dict) or fuse.get("type") != "noul" or not isinstance(fuse.get("noul"), (int, float)):
                raise ValueError(f"system_one_act_fuse_invalid:{question_id}")
            fuse_answers[question_id] = max(0.0, min(1.0, float(fuse["noul"])))
        return SystemOneActRecommendation(
            action=action,
            confidence=confidence,
            confident=confidence >= self.settings.route_confidence_threshold,
            rationale=actions[action],
            answers=fuse_answers,
            yes_threshold=self.settings.yes_threshold,
            model=str(response.get("model") or self.settings.model),
            usage=dict(response.get("usage") or {}),
            duration_ms=duration_ms,
        )

    def select_plan_tools(
        self,
        *,
        goal: str,
        phase: dict[str, Any],
        tools: tuple[Any, ...],
    ) -> SystemOneToolRegistrySelection:
        if not tools:
            return SystemOneToolRegistrySelection()
        registry = self._static_tool_registry(tools)
        state = {
            "goal": str(goal),
            "strategic_plan": phase,
        }
        started = monotonic()
        response = self.client.evaluate(state=state, questions=registry.questions)
        duration_ms = max(0, round((monotonic() - started) * 1000))
        selected: list[str] = []
        rejected: list[str] = []
        ambiguous: list[str] = []
        probabilities: dict[str, float] = {}
        for question_id, tool_id in registry.tool_ids_by_question.items():
            answer = response["answers"].get(question_id)
            if not isinstance(answer, dict) or answer.get("type") != "noul" or not isinstance(answer.get("noul"), (int, float)):
                raise ValueError(f"system_one_tool_relevance_answer_invalid:{question_id}")
            probability = max(0.0, min(1.0, float(answer["noul"])))
            probabilities[tool_id] = probability
            if probability >= self.settings.yes_threshold:
                selected.append(tool_id)
            elif probability <= self.settings.no_threshold:
                rejected.append(tool_id)
            else:
                ambiguous.append(tool_id)
        return SystemOneToolRegistrySelection(
            selected_tool_ids=tuple(selected),
            rejected_tool_ids=tuple(rejected),
            ambiguous_tool_ids=tuple(ambiguous),
            probabilities=probabilities,
            model=str(response.get("model") or self.settings.model),
            usage=dict(response.get("usage") or {}),
            duration_ms=duration_ms,
        )

    def evaluate_tactical_progress(
        self,
        *,
        goal: str,
        phase: dict[str, Any],
        subgoal: dict[str, Any],
        action: dict[str, Any],
        questions: list[dict[str, Any]] | None = None,
        execution_log: list[dict[str, Any]] | None = None,
    ) -> SystemOneTacticalReview:
        state = {
            "goal": str(goal),
            "strategic_plan": phase,
            "stage_goal": subgoal,
            "tool_execution_log": execution_log if execution_log is not None else [action],
        }
        supplied = list(questions or [])
        if not supplied:
            supplied = [{
                "question_id": "current_subgoal_complete",
                "type": "noul",
                "instructions": "Does the successful tool result semantically satisfy the current subgoal's declared completion condition?",
                "criteria": {
                    "true": "The observed result directly provides the required output and satisfies the declared completion condition.",
                    "false": "The call ran, but its result is irrelevant, incomplete, ambiguous, or does not establish the required output.",
                },
            }]
        acceptance_question_ids = {
            str(question.get("question_id") or f"acceptance_{index + 1}").strip()
            for index, question in enumerate(supplied)
        }
        supplied.extend(_TACTICAL_RETRY_FUSE_QUESTIONS)
        question_payload: dict[str, Any] = {}
        for index, question in enumerate(supplied):
            question_id = str(question.get("question_id") or f"acceptance_{index + 1}").strip()
            if not question_id or question_id in question_payload:
                raise ValueError("system_one_tactical_question_id_invalid")
            if question.get("type") != "noul" or not isinstance(question.get("criteria"), dict):
                raise ValueError(f"system_one_tactical_question_invalid:{question_id}")
            question_payload[question_id] = {
                "type": "noul",
                "instructions": str(question.get("instructions") or ""),
                "criteria": dict(question["criteria"]),
            }
        if not question_payload:
            raise ValueError("system_one_tactical_questions_required")
        started = monotonic()
        response = self.client.evaluate(state=state, questions=question_payload)
        duration_ms = max(0, round((monotonic() - started) * 1000))
        answers: dict[str, float] = {}
        for question_id in question_payload:
            answer = response["answers"].get(question_id)
            if not isinstance(answer, dict) or answer.get("type") != "noul" or not isinstance(answer.get("noul"), (int, float)):
                raise ValueError(f"system_one_tactical_review_invalid:{question_id}")
            answers[question_id] = max(0.0, min(1.0, float(answer["noul"])))
        acceptance_answers = {key: answers[key] for key in acceptance_question_ids}
        complete = all(probability >= self.settings.yes_threshold for probability in acceptance_answers.values())
        confident = all(
            probability >= self.settings.yes_threshold or probability <= self.settings.no_threshold
            for probability in acceptance_answers.values()
        )
        confidence = min(acceptance_answers.values()) if complete else min(1.0 - value for value in acceptance_answers.values())
        fuse_ids = {item["question_id"] for item in _TACTICAL_RETRY_FUSE_QUESTIONS}
        retry_approved = all(
            answers.get(question_id, 0.0) >= self.settings.yes_threshold
            for question_id in fuse_ids
        )
        return SystemOneTacticalReview(
            complete=complete,
            confidence=max(0.0, confidence),
            confident=confident,
            model=str(response.get("model") or self.settings.model),
            usage=dict(response.get("usage") or {}),
            duration_ms=duration_ms,
            answers=answers,
            retry_approved=retry_approved,
        )

    def _static_tool_registry(self, tools: tuple[Any, ...]) -> _StaticJevToolRegistry:
        signature = tuple(str(tool.tool_id) for tool in tools)
        if self._tool_registry is not None:
            if self._tool_registry.signature != signature:
                raise ValueError("system_one_static_tool_registry_changed_restart_required")
            return self._tool_registry
        questions: dict[str, Any] = {}
        tool_ids_by_question: dict[str, str] = {}
        native_questions = _load_jev_native_tool_registry()
        for tool in tools:
            tool_id = str(tool.tool_id)
            question_id = _tool_question_id(tool_id)
            if _tool_kind(tool) == "artifact":
                question = _artifact_jev_question(tool)
            else:
                question = native_questions.get(tool_id)
                if question is None:
                    raise ValueError(f"system_one_native_tool_template_missing:{tool_id}")
            questions[question_id] = json.loads(json.dumps(question, ensure_ascii=False))
            tool_ids_by_question[question_id] = tool_id
        self._tool_registry = _StaticJevToolRegistry(signature, questions, tool_ids_by_question)
        return self._tool_registry


def validate_and_save_system_one_settings(
    store: SQLiteSystemOneSettingsStore,
    *, values: dict[str, Any],
    transport: Transport | None = None,
) -> SystemOneSettings:
    current = store.get()
    supplied_key = str(values.get("api_key") or "").strip()
    api_key = "" if values.get("clear_api_key") is True else supplied_key or current.api_key
    candidate = _normalize_settings(SystemOneSettings(
        enabled=bool(values.get("enabled", current.enabled)),
        api_url=str(values.get("api_url") or current.api_url),
        model=str(values.get("model") or current.model),
        api_key=api_key,
        yes_threshold=float(values.get("yes_threshold", current.yes_threshold)),
        no_threshold=float(values.get("no_threshold", current.no_threshold)),
        route_confidence_threshold=float(values.get("route_confidence_threshold", current.route_confidence_threshold)),
    ))
    if not candidate.enabled:
        return store.save(candidate)
    client = TypeSafeSystemOneClient(
        api_url=candidate.api_url, api_key=candidate.api_key, model=candidate.model, transport=transport,
    )
    client.validate()
    return store.save(SystemOneSettings(
        **{**candidate.__dict__, "validated_at": _now(), "validation_error": ""}
    ))


def build_system_one_provider(settings: SystemOneSettings) -> JevCriterionDecisionProvider | None:
    if not settings.enabled or not settings.api_key or not settings.validated_at:
        return None
    return JevCriterionDecisionProvider(settings)


def _normalize_settings(settings: SystemOneSettings) -> SystemOneSettings:
    api_url = _normalize_url(settings.api_url)
    model = str(settings.model or "").strip() or DEFAULT_SYSTEM_ONE_MODEL
    yes = float(settings.yes_threshold)
    no = float(settings.no_threshold)
    route = float(settings.route_confidence_threshold)
    if not 0 <= no < yes <= 1:
        raise ValueError("system_one_thresholds_invalid")
    if not 0 <= route <= 1:
        raise ValueError("system_one_route_threshold_invalid")
    return SystemOneSettings(
        enabled=bool(settings.enabled), api_url=api_url, model=model,
        api_key=str(settings.api_key or "").strip(), yes_threshold=yes, no_threshold=no,
        route_confidence_threshold=route, validated_at=str(settings.validated_at or ""),
        validation_error=str(settings.validation_error or ""), updated_at=str(settings.updated_at or ""),
    )


def _normalize_url(value: str) -> str:
    url = str(value or "").strip().rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("system_one_api_url_invalid")
    return url


def _bounded_evidence(raw_entries: Any) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for raw in list(raw_entries or [])[-12:]:
        if not isinstance(raw, dict) or str(raw.get("status") or "") != "success":
            continue
        evidence_ref = str(raw.get("evidence_ref") or "").strip()
        if not evidence_ref:
            continue
        rendered = json.dumps(raw.get("result"), ensure_ascii=False, sort_keys=True, default=str)
        entries.append({"evidence_ref": evidence_ref, "summary": rendered[:2000]})
    return entries


def _bounded_json(value: Any, limit: int) -> Any:
    rendered = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    if len(rendered) <= limit:
        return value
    return {"truncated_json": rendered[:limit]}


def _tool_question_id(tool_id: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(tool_id or "tool")).strip("_").lower()[:48] or "tool"
    digest = hashlib.sha256(str(tool_id).encode("utf-8")).hexdigest()[:10]
    return f"tool_relevance__{slug}__{digest}"


@lru_cache(maxsize=1)
def _load_jev_native_tool_registry() -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(_JEV_TOOL_REGISTRY_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError("system_one_native_tool_template_invalid") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("system_one_native_tool_template_invalid")
    raw_questions = payload.get("questions")
    if not isinstance(raw_questions, dict) or not raw_questions:
        raise ValueError("system_one_native_tool_template_invalid")
    questions: dict[str, dict[str, Any]] = {}
    for tool_id, question in raw_questions.items():
        normalized_id = str(tool_id or "").strip()
        if not normalized_id.startswith("native.") or not _valid_jev_tool_question(question):
            raise ValueError(f"system_one_native_tool_template_invalid:{normalized_id or '(missing)'}")
        questions[normalized_id] = dict(question)
    return questions


def _valid_jev_tool_question(value: Any) -> bool:
    if not isinstance(value, dict) or value.get("type") != "noul":
        return False
    if not str(value.get("instructions") or "").strip():
        return False
    criteria = value.get("criteria")
    return (
        isinstance(criteria, dict)
        and bool(str(criteria.get("true") or "").strip())
        and bool(str(criteria.get("false") or "").strip())
    )


def _tool_kind(tool: Any) -> str:
    kind = getattr(tool, "kind", "")
    return str(getattr(kind, "value", kind) or "").strip().lower()


def _artifact_jev_question(tool: Any) -> dict[str, Any]:
    name = _jev_tool_name(tool)
    description = _jev_tool_description(tool)
    return {
        "type": "noul",
        "instructions": f"Does this phase need {name}?",
        "criteria": {
            "true": f"The phase needs {name}: {description}",
            "false": f"The phase does not need {name}.",
        },
    }


def _jev_tool_name(tool: Any) -> str:
    name = str(getattr(tool, "name", "") or getattr(tool, "tool_id", "") or "this tool").strip()
    return name.replace("_", " ")[:200]


def _jev_tool_description(tool: Any) -> str:
    description = str(getattr(tool, "description", "") or "No description supplied.").strip()[:1200]
    return description if description.endswith((".", "!", "?")) else description + "."


def _http_transport(url: str, api_key: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except HTTPError as exc:
        if exc.code == 401:
            raise ValueError("system_one_api_key_invalid") from exc
        if exc.code == 422:
            raise ValueError("system_one_request_invalid") from exc
        if exc.code in {429, 529}:
            raise RuntimeError(f"system_one_temporarily_unavailable:{exc.code}") from exc
        raise RuntimeError(f"system_one_http_error:{exc.code}") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError("system_one_connection_failed") from exc
    try:
        value = json.loads(body)
    except ValueError as exc:
        raise ValueError("system_one_response_invalid") from exc
    if not isinstance(value, dict):
        raise ValueError("system_one_response_invalid")
    return value


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class _ConnectionProxy:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
    def __enter__(self) -> sqlite3.Connection:
        return self.connection
    def __exit__(self, exc_type, exc, traceback) -> bool:
        if exc_type is None:
            self.connection.commit()
        else:
            self.connection.rollback()
        return False
