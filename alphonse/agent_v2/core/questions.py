"""V2-native question interrupts and parking store."""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal

from alphonse.agent_v2.core.intelligence.task_state import TaskState

QuestionKind = Literal["open_text", "yes_no", "single_choice"]
QuestionStatus = Literal["pending", "answered", "expired", "cancelled"]

DEFAULT_EXPIRES_IN_SECONDS = 24 * 60 * 60


@dataclass(frozen=True)
class QuestionChoice:
    id: str
    label: str

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "label": self.label}


@dataclass(frozen=True)
class QuestionInterrupt:
    question_id: str
    task_id: str
    run_id: str
    thread_id: str
    respondent_user_id: str
    originator_user_id: str
    message: str
    kind: QuestionKind
    choices: tuple[QuestionChoice, ...] = ()
    status: QuestionStatus = "pending"
    expires_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def response_schema(self) -> dict[str, Any]:
        if self.kind == "yes_no":
            return {
                "type": "object",
                "additionalProperties": False,
                "properties": {"answer": {"type": "boolean"}},
                "required": ["answer"],
            }
        if self.kind == "single_choice":
            return {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "choice_id": {
                        "type": "string",
                        "enum": [choice.id for choice in self.choices],
                    }
                },
                "required": ["choice_id"],
            }
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {"text": {"type": "string", "minLength": 1}},
            "required": ["text"],
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "task_id": self.task_id,
            "run_id": self.run_id,
            "thread_id": self.thread_id,
            "respondent_user_id": self.respondent_user_id,
            "originator_user_id": self.originator_user_id,
            "message": self.message,
            "kind": self.kind,
            "choices": [choice.to_dict() for choice in self.choices],
            "response_schema": self.response_schema,
            "status": self.status,
            "expires_at": self.expires_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "QuestionInterrupt":
        return cls(
            question_id=str(value.get("question_id") or "").strip(),
            task_id=str(value.get("task_id") or "").strip(),
            run_id=str(value.get("run_id") or "").strip(),
            thread_id=str(value.get("thread_id") or "").strip(),
            respondent_user_id=str(value.get("respondent_user_id") or "").strip(),
            originator_user_id=str(value.get("originator_user_id") or "").strip(),
            message=str(value.get("message") or "").strip(),
            kind=_normalize_kind(value.get("kind")),
            choices=tuple(_normalize_choices(value.get("choices"), require_for_choice=False)),
            status=_normalize_status(value.get("status")),
            expires_at=str(value.get("expires_at") or "").strip(),
            metadata=dict(value.get("metadata")) if isinstance(value.get("metadata"), dict) else {},
        )


@dataclass(frozen=True)
class QuestionAnswerResult:
    handled: bool
    question: QuestionInterrupt | None = None
    resumed_task: TaskState | None = None
    answer: dict[str, Any] | None = None
    message: str = ""
    ambiguous: bool = False
    invalid: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "handled": self.handled,
            "question_id": self.question.question_id if self.question is not None else None,
            "task_id": self.question.task_id if self.question is not None else None,
            "answer": dict(self.answer or {}),
            "message": self.message,
            "ambiguous": self.ambiguous,
            "invalid": self.invalid,
            "resumed_task": self.resumed_task.to_dict() if self.resumed_task is not None else None,
        }


class SQLiteQuestionStore:
    """SQLite-backed v2 question and parked-task store."""

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self.db_path = str(db_path)
        self._memory_connection: sqlite3.Connection | None = None
        if self.db_path == ":memory:":
            self._memory_connection = sqlite3.connect(":memory:", check_same_thread=False)
            self._memory_connection.row_factory = sqlite3.Row
        self._ensure_schema()

    @classmethod
    def default(cls) -> "SQLiteQuestionStore":
        return cls(_default_db_path())

    def create_question(
        self,
        *,
        task: TaskState,
        question: str,
        kind: str = "open_text",
        choices: Any = None,
        respondent_user_id: str | None = None,
        expires_in_seconds: int | None = None,
        delivery_metadata: dict[str, Any] | None = None,
    ) -> QuestionInterrupt:
        message = str(question or "").strip()
        if not message:
            raise ValueError("question_required")

        normalized_kind = _normalize_kind(kind)
        normalized_choices = tuple(_normalize_choices(choices, require_for_choice=normalized_kind == "single_choice"))
        task_id = _ensure_task_id(task)
        originator = str(task.user or "").strip() or "unknown"
        respondent = str(respondent_user_id or originator).strip() or originator
        run_id = str(task.metadata.get("run_id") or task.message_id or task_id).strip() or task_id
        thread_id = str(task.project_id or task.correlation_id or task_id).strip() or task_id
        now = _now()
        ttl = max(int(expires_in_seconds or DEFAULT_EXPIRES_IN_SECONDS), 60)
        child_task_id = str(uuid.uuid4()) if respondent != originator else ""
        interrupt = QuestionInterrupt(
            question_id=str(uuid.uuid4()),
            task_id=task_id,
            run_id=run_id,
            thread_id=thread_id,
            respondent_user_id=respondent,
            originator_user_id=originator,
            message=message,
            kind=normalized_kind,
            choices=normalized_choices,
            status="pending",
            expires_at=(now + timedelta(seconds=ttl)).isoformat(),
            metadata={
                "delivery": dict(delivery_metadata or {}),
                "child_task_id": child_task_id,
            },
        )

        task.status = "waiting_user"
        task.metadata["pending_question_id"] = interrupt.question_id
        task.metadata["question_interrupt"] = interrupt.to_dict()
        task.append_update(f"Task parked waiting for answer to question {interrupt.question_id}.")
        self.save_task_checkpoint(task, status="waiting_user")

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO v2_questions (
                  question_id, task_id, run_id, thread_id, respondent_user_id,
                  originator_user_id, message, kind, choices_json, response_schema_json,
                  status, expires_at, answer_json, delivery_metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, NULL, ?, ?, ?)
                """,
                (
                    interrupt.question_id,
                    interrupt.task_id,
                    interrupt.run_id,
                    interrupt.thread_id,
                    interrupt.respondent_user_id,
                    interrupt.originator_user_id,
                    interrupt.message,
                    interrupt.kind,
                    json.dumps([choice.to_dict() for choice in interrupt.choices], sort_keys=True),
                    json.dumps(interrupt.response_schema, sort_keys=True),
                    interrupt.expires_at,
                    json.dumps(dict(delivery_metadata or {}), sort_keys=True),
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
            if child_task_id:
                conn.execute(
                    """
                    INSERT INTO v2_task_dependencies (
                      parent_task_id, child_task_id, question_id, status, created_at, updated_at
                    ) VALUES (?, ?, ?, 'pending', ?, ?)
                    """,
                    (task_id, child_task_id, interrupt.question_id, now.isoformat(), now.isoformat()),
                )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO v2_task_checkpoints (
                      task_id, task_state_json, status, owner_id, project_id, correlation_id,
                      version, created_at, updated_at
                    ) VALUES (?, ?, 'waiting_user', ?, ?, ?, 1, ?, ?)
                    """,
                    (
                        child_task_id,
                        json.dumps(
                            {
                                "task_id": child_task_id,
                                "user": respondent,
                                "goal": message,
                                "status": "waiting_user",
                                "metadata": {"parent_task_id": task_id, "question_id": interrupt.question_id},
                            },
                            sort_keys=True,
                        ),
                        respondent,
                        task.project_id,
                        task.correlation_id,
                        now.isoformat(),
                        now.isoformat(),
                    ),
                )
        return interrupt

    def bind_delivery_metadata(self, *, question_id: str, metadata: dict[str, Any]) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT delivery_metadata_json FROM v2_questions WHERE question_id = ?",
                (str(question_id or "").strip(),),
            ).fetchone()
            if row is None:
                return False
            existing = _json_object(row["delivery_metadata_json"])
            existing.update(dict(metadata or {}))
            cursor = conn.execute(
                "UPDATE v2_questions SET delivery_metadata_json = ?, updated_at = ? WHERE question_id = ?",
                (json.dumps(existing, sort_keys=True), _now().isoformat(), str(question_id or "").strip()),
            )
            return cursor.rowcount == 1

    def save_task_checkpoint(self, task: TaskState, *, status: str | None = None) -> int:
        task_id = _ensure_task_id(task)
        now = _now().isoformat()
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT version, created_at FROM v2_task_checkpoints WHERE task_id = ?",
                (task_id,),
            ).fetchone()
            version = int(existing["version"]) + 1 if existing is not None else 1
            created_at = str(existing["created_at"]) if existing is not None else now
            conn.execute(
                """
                INSERT OR REPLACE INTO v2_task_checkpoints (
                  task_id, task_state_json, status, owner_id, project_id, correlation_id,
                  version, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    task_id,
                    json.dumps(task.to_dict(), sort_keys=True),
                    str(status or task.status or "running"),
                    str(task.user or ""),
                    task.project_id,
                    task.correlation_id,
                    version,
                    created_at,
                    now,
                ),
            )
            return version

    def load_task_checkpoint(self, task_id: str) -> TaskState | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT task_state_json FROM v2_task_checkpoints WHERE task_id = ?",
                (str(task_id or "").strip(),),
            ).fetchone()
        if row is None:
            return None
        return TaskState.from_dict(_json_object(row["task_state_json"]))

    def get_question(self, question_id: str) -> QuestionInterrupt | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM v2_questions WHERE question_id = ?",
                (str(question_id or "").strip(),),
            ).fetchone()
        return _question_from_row(row) if row is not None else None

    def list_pending_for_respondent(self, respondent_user_id: str) -> list[QuestionInterrupt]:
        self.expire_questions()
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT * FROM v2_questions
                WHERE respondent_user_id = ? AND status = 'pending'
                ORDER BY created_at ASC
                """,
                (str(respondent_user_id or "").strip(),),
            ).fetchall()
        return [_question_from_row(row) for row in rows]

    def route_answer(
        self,
        *,
        respondent_user_id: str,
        text: str | None = None,
        payload: dict[str, Any] | None = None,
        question_id: str | None = None,
        reply_to_provider_message_id: str | None = None,
    ) -> QuestionAnswerResult:
        respondent = str(respondent_user_id or "").strip()
        if not respondent:
            return QuestionAnswerResult(handled=False)
        question = self._select_question(
            respondent_user_id=respondent,
            question_id=question_id,
            reply_to_provider_message_id=reply_to_provider_message_id,
        )
        if question == "ambiguous":
            return QuestionAnswerResult(
                handled=True,
                ambiguous=True,
                message="You have more than one pending question. Please answer the specific question.",
            )
        if question is None:
            return QuestionAnswerResult(handled=False)

        answer = _normalize_answer(question, text=text, payload=payload)
        if answer is None:
            return QuestionAnswerResult(
                handled=True,
                question=question,
                invalid=True,
                message=_invalid_answer_message(question),
            )

        task = self.load_task_checkpoint(question.task_id)
        if task is None:
            return QuestionAnswerResult(handled=True, question=question, answer=answer, message="Question answered.")

        task.status = "running"
        task.metadata.pop("pending_question_id", None)
        task.metadata.pop("question_answer", None)
        task.append_conversation_message(question.respondent_user_id, _answer_text(answer))
        _record_question_answer_tool_result(task=task, question=question, answer=answer)
        task.append_update(f"Question {question.question_id} answered; task resumed.")

        now = _now().isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                UPDATE v2_questions
                SET status = 'answered', answer_json = ?, updated_at = ?
                WHERE question_id = ? AND respondent_user_id = ? AND status = 'pending'
                """,
                (
                    json.dumps(answer, sort_keys=True),
                    now,
                    question.question_id,
                    respondent,
                ),
            )
            if cursor.rowcount != 1:
                return QuestionAnswerResult(handled=False)
            conn.execute(
                """
                UPDATE v2_task_dependencies
                SET status = 'answered', updated_at = ?
                WHERE question_id = ? AND status = 'pending'
                """,
                (now, question.question_id),
            )
        self.save_task_checkpoint(task, status="queued")
        answered = QuestionInterrupt.from_dict({**question.to_dict(), "status": "answered"})
        return QuestionAnswerResult(
            handled=True,
            question=answered,
            resumed_task=task,
            answer=answer,
            message="Question answered; task resumed.",
        )

    def cancel_question(self, question_id: str) -> bool:
        now = _now().isoformat()
        with self._connect() as conn:
            cursor = conn.execute(
                "UPDATE v2_questions SET status = 'cancelled', updated_at = ? WHERE question_id = ? AND status = 'pending'",
                (now, str(question_id or "").strip()),
            )
            if cursor.rowcount == 1:
                conn.execute(
                    "UPDATE v2_task_dependencies SET status = 'cancelled', updated_at = ? WHERE question_id = ?",
                    (now, str(question_id or "").strip()),
                )
            return cursor.rowcount == 1

    def expire_questions(self, *, now: str | None = None) -> list[QuestionInterrupt]:
        current = str(now or _now().isoformat()).strip()
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM v2_questions WHERE status = 'pending' AND expires_at <= ?",
                (current,),
            ).fetchall()
            conn.execute(
                "UPDATE v2_questions SET status = 'expired', updated_at = ? WHERE status = 'pending' AND expires_at <= ?",
                (current, current),
            )
        return [_question_from_row(row) for row in rows]

    def _select_question(
        self,
        *,
        respondent_user_id: str,
        question_id: str | None,
        reply_to_provider_message_id: str | None,
    ) -> QuestionInterrupt | str | None:
        self.expire_questions()
        qid = str(question_id or "").strip()
        if qid:
            question = self.get_question(qid)
            if question is None or question.status != "pending" or question.respondent_user_id != respondent_user_id:
                return None
            return question

        reply_id = str(reply_to_provider_message_id or "").strip()
        if reply_id:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT * FROM v2_questions
                    WHERE respondent_user_id = ? AND status = 'pending'
                      AND json_extract(delivery_metadata_json, '$.provider_message_id') = ?
                    ORDER BY created_at DESC
                    LIMIT 1
                    """,
                    (respondent_user_id, reply_id),
                ).fetchone()
            if row is not None:
                return _question_from_row(row)

        pending = self.list_pending_for_respondent(respondent_user_id)
        if len(pending) == 1:
            return pending[0]
        if len(pending) > 1:
            return "ambiguous"
        return None

    def _connect(self) -> sqlite3.Connection:
        if self._memory_connection is not None:
            return _ConnectionProxy(self._memory_connection)
        path = Path(self.db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS v2_task_checkpoints (
                  task_id TEXT PRIMARY KEY,
                  task_state_json TEXT NOT NULL,
                  status TEXT NOT NULL,
                  owner_id TEXT,
                  project_id TEXT,
                  correlation_id TEXT,
                  version INTEGER NOT NULL DEFAULT 1,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  CHECK (status IN ('queued', 'running', 'waiting_user', 'done', 'failed', 'cancelled'))
                ) STRICT;

                CREATE TABLE IF NOT EXISTS v2_questions (
                  question_id TEXT PRIMARY KEY,
                  task_id TEXT NOT NULL,
                  run_id TEXT NOT NULL,
                  thread_id TEXT NOT NULL,
                  respondent_user_id TEXT NOT NULL,
                  originator_user_id TEXT NOT NULL,
                  message TEXT NOT NULL,
                  kind TEXT NOT NULL,
                  choices_json TEXT NOT NULL,
                  response_schema_json TEXT NOT NULL,
                  status TEXT NOT NULL,
                  expires_at TEXT NOT NULL,
                  answer_json TEXT,
                  delivery_metadata_json TEXT NOT NULL DEFAULT '{}',
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  CHECK (kind IN ('open_text', 'yes_no', 'single_choice')),
                  CHECK (status IN ('pending', 'answered', 'expired', 'cancelled'))
                ) STRICT;

                CREATE INDEX IF NOT EXISTS idx_v2_questions_respondent_status
                  ON v2_questions (respondent_user_id, status, created_at);

                CREATE TABLE IF NOT EXISTS v2_task_dependencies (
                  parent_task_id TEXT NOT NULL,
                  child_task_id TEXT NOT NULL,
                  question_id TEXT NOT NULL,
                  status TEXT NOT NULL,
                  created_at TEXT NOT NULL,
                  updated_at TEXT NOT NULL,
                  PRIMARY KEY (parent_task_id, child_task_id, question_id),
                  CHECK (status IN ('pending', 'answered', 'cancelled', 'expired'))
                ) STRICT;
                """
            )


class _ConnectionProxy:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def __enter__(self) -> sqlite3.Connection:
        return self._connection

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if exc_type is None:
            self._connection.commit()
        else:
            self._connection.rollback()


def _question_from_row(row: sqlite3.Row) -> QuestionInterrupt:
    return QuestionInterrupt(
        question_id=str(row["question_id"]),
        task_id=str(row["task_id"]),
        run_id=str(row["run_id"]),
        thread_id=str(row["thread_id"]),
        respondent_user_id=str(row["respondent_user_id"]),
        originator_user_id=str(row["originator_user_id"]),
        message=str(row["message"]),
        kind=_normalize_kind(row["kind"]),
        choices=tuple(_normalize_choices(_json_list(row["choices_json"]), require_for_choice=False)),
        status=_normalize_status(row["status"]),
        expires_at=str(row["expires_at"]),
        metadata={"delivery": _json_object(row["delivery_metadata_json"])},
    )


def _record_question_answer_tool_result(
    *,
    task: TaskState,
    question: QuestionInterrupt,
    answer: dict[str, Any],
) -> None:
    calls = _json_list(task.plan_json)
    if not calls:
        calls = []
    result = {
        "question_id": question.question_id,
        "question_kind": question.kind,
        "answer": dict(answer),
        "answered_by": question.respondent_user_id,
    }
    now = _now().isoformat()
    for index in range(len(calls) - 1, -1, -1):
        call = calls[index]
        if not isinstance(call, dict):
            continue
        if str(call.get("tool_id") or "").strip() != "native.ask_question":
            continue
        execution = call.get("execution")
        if not isinstance(execution, dict):
            continue
        execution_result = execution.get("result")
        if not _execution_matches_question(execution_result, question.question_id):
            continue
        updated_execution = dict(execution)
        updated_execution["status"] = "success"
        updated_execution["result"] = result
        updated_execution["exception"] = ""
        updated_execution["finished_at"] = now
        updated_call = dict(call)
        updated_call["execution"] = updated_execution
        calls[index] = updated_call
        task.plan_json = json.dumps(calls, indent=2, sort_keys=True)
        return

    calls.append(
        {
            "id": f"question-answer-{question.question_id}",
            "tool_id": "native.ask_question",
            "tool_name": "ask_question",
            "arguments": {
                "question": question.message,
                "question_kind": question.kind,
            },
            "internal_state": "Question answer received.",
            "execution": {
                "status": "success",
                "result": result,
                "exception": "",
                "started_at": now,
                "finished_at": now,
            },
        }
    )
    task.plan_json = json.dumps(calls, indent=2, sort_keys=True)


def _execution_matches_question(value: Any, question_id: str) -> bool:
    if not isinstance(value, dict):
        return False
    if str(value.get("question_id") or "").strip() == question_id:
        return True
    interrupt = value.get("question_interrupt")
    if isinstance(interrupt, dict) and str(interrupt.get("question_id") or "").strip() == question_id:
        return True
    return False


def _ensure_task_id(task: TaskState) -> str:
    task_id = str(task.task_id or "").strip()
    if not task_id:
        task_id = str(uuid.uuid4())
        task.task_id = task_id
    return task_id


def _normalize_kind(value: Any) -> QuestionKind:
    rendered = str(value or "open_text").strip().lower()
    if rendered not in {"open_text", "yes_no", "single_choice"}:
        raise ValueError(f"invalid_question_kind: {rendered}")
    return rendered  # type: ignore[return-value]


def _normalize_status(value: Any) -> QuestionStatus:
    rendered = str(value or "pending").strip().lower()
    if rendered not in {"pending", "answered", "expired", "cancelled"}:
        raise ValueError(f"invalid_question_status: {rendered}")
    return rendered  # type: ignore[return-value]


def _normalize_choices(value: Any, *, require_for_choice: bool) -> list[QuestionChoice]:
    if value is None:
        if require_for_choice:
            raise ValueError("single_choice_choices_required")
        return []
    if not isinstance(value, list):
        raise ValueError("question_choices_must_be_list")
    choices: list[QuestionChoice] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("question_choice_must_be_object")
        choice_id = str(item.get("id") or "").strip()
        label = str(item.get("label") or "").strip()
        if not choice_id or not label:
            raise ValueError("question_choice_id_and_label_required")
        choices.append(QuestionChoice(id=choice_id, label=label))
    if require_for_choice and not choices:
        raise ValueError("single_choice_choices_required")
    return choices


def _normalize_answer(
    question: QuestionInterrupt,
    *,
    text: str | None,
    payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    value = dict(payload or {})
    raw_text = str(text or "").strip()
    if question.kind == "open_text":
        answer_text = str(value.get("text") or raw_text).strip()
        return {"text": answer_text} if answer_text else None
    if question.kind == "yes_no":
        raw_answer = value.get("answer")
        if isinstance(raw_answer, bool):
            return {"answer": raw_answer}
        lowered = str(raw_answer if raw_answer is not None else raw_text).strip().lower()
        if lowered in {"yes", "y", "true", "1", "si", "sí"}:
            return {"answer": True}
        if lowered in {"no", "n", "false", "0"}:
            return {"answer": False}
        return None
    raw_choice = str(value.get("choice_id") or raw_text).strip()
    for choice in question.choices:
        if raw_choice == choice.id or raw_choice.lower() == choice.label.lower():
            return {"choice_id": choice.id, "label": choice.label}
    return None


def _invalid_answer_message(question: QuestionInterrupt) -> str:
    if question.kind == "yes_no":
        return "Please answer yes or no."
    if question.kind == "single_choice":
        options = ", ".join(choice.label for choice in question.choices)
        return f"Please choose one of: {options}."
    return "Please provide a non-empty answer."


def _answer_text(answer: dict[str, Any]) -> str:
    if "text" in answer:
        return str(answer.get("text") or "")
    if "answer" in answer:
        return "yes" if bool(answer.get("answer")) else "no"
    if "label" in answer:
        return str(answer.get("label") or "")
    return json.dumps(answer, sort_keys=True)


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    try:
        parsed = json.loads(str(value or "{}"))
    except ValueError:
        return {}
    return dict(parsed) if isinstance(parsed, dict) else {}


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    try:
        parsed = json.loads(str(value or "[]"))
    except ValueError:
        return []
    return list(parsed) if isinstance(parsed, list) else []


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _default_db_path() -> str:
    configured = (
        os.getenv("ALPHONSE_V2_QUESTION_DB_PATH")
        or os.getenv("ALPHONSE_V2_DB_PATH")
        or os.getenv("NERVE_DB_PATH")
    )
    if configured:
        return configured
    return str(Path.home() / ".alphonse" / "v2-questions.sqlite3")
