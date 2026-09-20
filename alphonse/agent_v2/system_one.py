"""TypeSafe Jev configuration, HTTP client, and bounded criterion decisions."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import monotonic
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from alphonse.agent_v2.database import connect_database, default_database_path

DEFAULT_SYSTEM_ONE_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_SYSTEM_ONE_MODEL = "jev-latest"


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


class JevCriterionDecisionProvider:
    def __init__(self, settings: SystemOneSettings, *, transport: Transport | None = None) -> None:
        self.settings = _normalize_settings(settings)
        self.client = TypeSafeSystemOneClient(
            api_url=self.settings.api_url, api_key=self.settings.api_key,
            model=self.settings.model, transport=transport,
        )

    def evaluate(self, *, contract: dict[str, Any], phase: dict[str, Any], evidence: dict[str, Any]) -> SystemOneReviewResult:
        criteria = [
            item for item in contract.get("criteria") or []
            if isinstance(item, dict) and item.get("superseded") is not True
            and item.get("required", True) and item.get("verification") != "system"
            and item.get("status") != "satisfied"
        ]
        entries = _bounded_evidence(evidence.get("entries") if isinstance(evidence, dict) else [])
        state = {
            "acceptance_criteria": {str(item.get("id")): str(item.get("statement")) for item in criteria},
            "phase": {"phase_id": phase.get("phase_id"), "objective": phase.get("objective")},
            "verified_evidence": {item["evidence_ref"]: item["summary"] for item in entries},
        }
        questions: dict[str, Any] = {
            "recommended_route": {
                "type": "choice",
                "instructions": "Which route is justified by the current phase outcome and verified evidence?",
                "criteria": {
                    "complete": "Every required acceptance criterion has direct supporting evidence.",
                    "continue": "Evidence is valid but one or more required outcomes still need work.",
                    "replan": "The current strategy cannot safely or effectively complete the remaining work.",
                    "ask_user": "Progress requires information or a decision only the user can provide.",
                    "fail": "The task cannot be completed safely within the authorized scope.",
                },
            }
        }
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
        route = answers.get("recommended_route")
        if not isinstance(route, dict) or route.get("type") != "choice":
            raise ValueError("system_one_route_answer_invalid")
        route_name = str(route.get("choice") or "")
        route_probability = float((route.get("probabilities") or {}).get(route_name) or 0.0)
        return SystemOneReviewResult(
            updates=tuple(updates), ambiguous_criterion_ids=tuple(ambiguous),
            recommended_route=route_name, route_confidence=route_probability,
            route_confident=(route_probability >= self.settings.route_confidence_threshold),
            model=str(response.get("model") or self.settings.model),
            usage=dict(response.get("usage") or {}), duration_ms=duration_ms,
        )


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
