"""Versioned, task-scoped budget accounting for a future execution engine.

This module performs no I/O or provider calls. A caller must atomically persist
and fence every reservation *before* dispatch, and persist every settlement.
A revision enables compare-and-swap; copying this object to two workers without
an atomic check does not provide concurrency safety. Unknown or in-flight work
is never automatically refunded after a crash, timeout, or restart.

Elapsed time is wall time from task creation, including user/provider waits and
process downtime, except for explicitly attested continuation pauses. Clocks
are injected by the caller as integer UTC milliseconds.
Budget decisions are not authorization, cancellation, or effect verification.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Mapping
from types import MappingProxyType

SCHEMA_VERSION = 1
CONTINUATION_SCHEMA_VERSION = 2
_MAX_COUNTER = (1 << 63) - 1
DIMENSIONS = ("actions", "model_calls", "input_tokens", "output_tokens")
TOKEN_DIMENSIONS = ("input_tokens", "output_tokens")


def _integer(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"budget_invalid:{name}")
    return value


def _vector(value: Mapping[str, int], name: str) -> dict[str, int]:
    if not isinstance(value, Mapping) or set(value) != set(DIMENSIONS):
        raise ValueError(f"budget_invalid:{name}")
    return {key: _integer(value[key], f"{name}.{key}") for key in DIMENSIONS}


def _now(value: int, started: int) -> int:
    value = _integer(value, "now_ms")
    if value < started:
        raise ValueError("budget_clock_before_start")
    return value


class BudgetExhausted(ValueError):
    """The caller must stop work or use its bounded final-response path."""


@dataclass(frozen=True)
class BudgetLimits:
    total: Mapping[str, int]
    final_reserve: Mapping[str, int]
    elapsed_ms: int
    final_reserve_ms: int

    def __post_init__(self) -> None:
        total = _vector(self.total, "total")
        final = _vector(self.final_reserve, "final_reserve")
        elapsed = _integer(self.elapsed_ms, "elapsed_ms")
        final_ms = _integer(self.final_reserve_ms, "final_reserve_ms")
        if any(final[k] > total[k] for k in DIMENSIONS) or final_ms > elapsed:
            raise ValueError("budget_final_reserve_exceeds_total")
        object.__setattr__(self, "total", MappingProxyType(total))
        object.__setattr__(self, "final_reserve", MappingProxyType(final))

    def to_dict(self) -> dict[str, Any]:
        return {"total": dict(self.total), "final_reserve": dict(self.final_reserve),
                "elapsed_ms": self.elapsed_ms, "final_reserve_ms": self.final_reserve_ms}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BudgetLimits":
        if not isinstance(value, Mapping) or set(value) != {"total", "final_reserve", "elapsed_ms", "final_reserve_ms"}:
            raise ValueError("budget_invalid:limits")
        return cls(**value)


class TaskBudget:
    """An accounting snapshot; caller persists with revision-based CAS.

    `work` reservations can consume only total minus final reserve. Exactly one
    `final` model request may use the remaining total. If that cannot be made,
    the caller must use a deterministic, no-model fallback. Unresolved or unknown
    model usage blocks even the final model request until reconciled. All provider
    attempts, including retries and failures, need distinct reservation IDs.
    """

    def __init__(self, limits: BudgetLimits, *, started_at_ms: int) -> None:
        if not isinstance(limits, BudgetLimits):
            raise ValueError("budget_invalid:limits")
        self.limits = limits
        self.started_at_ms = _integer(started_at_ms, "started_at_ms")
        self.revision = 0
        self.last_observed_at_ms = self.started_at_ms
        self._reservations: dict[str, dict[str, Any]] = {}
        self._continuation_windows: list[dict[str, Any]] = []
        self._excluded_pause_ms = 0

    def to_dict(self) -> dict[str, Any]:
        result = {"schema_version": CONTINUATION_SCHEMA_VERSION if self._continuation_windows else SCHEMA_VERSION, "revision": self.revision,
                "started_at_ms": self.started_at_ms, "last_observed_at_ms": self.last_observed_at_ms, "limits": self.limits.to_dict(),
                "reservations": deepcopy(self._reservations)}
        if self._continuation_windows:
            result.update(continuation_windows=deepcopy(self._continuation_windows), excluded_pause_ms=self._excluded_pause_ms)
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskBudget":
        base = {"schema_version", "revision", "started_at_ms", "last_observed_at_ms", "limits", "reservations"}
        if not isinstance(value, Mapping):
            raise ValueError("budget_invalid:snapshot")
        version = value.get("schema_version")
        if type(version) is not int or version not in {SCHEMA_VERSION, CONTINUATION_SCHEMA_VERSION}:
            raise ValueError("budget_schema_unsupported")
        if set(value) != base | ({"continuation_windows", "excluded_pause_ms"} if version == CONTINUATION_SCHEMA_VERSION else set()):
            raise ValueError("budget_invalid:snapshot")
        obj = cls(BudgetLimits.from_dict(value["limits"]), started_at_ms=value["started_at_ms"])
        obj.revision = _integer(value["revision"], "revision")
        obj.last_observed_at_ms = _now(value["last_observed_at_ms"], obj.started_at_ms)
        rows = value["reservations"]
        if not isinstance(rows, dict):
            raise ValueError("budget_invalid:reservations")
        for key, row in rows.items():
            if not isinstance(key, str) or not key.strip() or not isinstance(row, dict):
                raise ValueError("budget_invalid:reservation")
            if set(row) != {"lane", "kind", "reserved", "status", "actual", "unknown_tokens", "outcome"}:
                raise ValueError("budget_invalid:reservation")
            if row["lane"] not in {"work", "final"} or row["kind"] not in {"tool", "model"}:
                raise ValueError("budget_invalid:reservation_type")
            if row["lane"] == "final" and row["kind"] != "model":
                raise ValueError("budget_invalid:final_kind")
            reserved = _vector(row["reserved"], "reserved")
            cls._validate_request(row["kind"], reserved)
            if row["status"] not in {"reserved", "settled", "released"}:
                raise ValueError("budget_invalid:reservation_status")
            unknown = row["unknown_tokens"]
            if not isinstance(unknown, list) or len(set(unknown)) != len(unknown) or any(x not in TOKEN_DIMENSIONS for x in unknown):
                raise ValueError("budget_invalid:unknown_tokens")
            actual = row["actual"]
            if row["status"] == "settled":
                actual = _vector(actual, "actual")
                if row["outcome"] not in {"success", "failure", "timeout", "cancelled"}:
                    raise ValueError("budget_invalid:outcome")
                if any(actual[x] != reserved[x] for x in ("actions", "model_calls")):
                    raise ValueError("budget_invalid:actual_counts")
                if any(actual[x] != reserved[x] for x in unknown):
                    raise ValueError("budget_invalid:unknown_charge")
                if row["kind"] == "tool" and unknown:
                    raise ValueError("budget_invalid:unknown_tokens")
            elif actual is not None or unknown or row["outcome"] is not None:
                raise ValueError("budget_invalid:unsettled_fields")
            obj._reservations[key] = deepcopy(row)
        if sum(row["lane"] == "final" and row["status"] != "released" for row in rows.values()) > 1:
            raise ValueError("budget_invalid:multiple_final_attempts")
        if obj.revision < len(rows):
            raise ValueError("budget_invalid:revision")
        if version == CONTINUATION_SCHEMA_VERSION:
            windows = value['continuation_windows']
            if not isinstance(windows, list) or not windows:
                raise ValueError('budget_invalid:continuation_windows')
            previous_end = obj.started_at_ms
            grants = set()
            excluded = 0
            added = {key: 0 for key in DIMENSIONS}
            added_elapsed = 0
            for window in windows:
                if (not isinstance(window, dict) or set(window) != {'grant_id', 'pause_started_ms', 'resumed_ms', 'increment', 'elapsed_increment_ms'}
                        or not isinstance(window['grant_id'], str) or not window['grant_id'].strip()
                        or window['grant_id'] in grants):
                    raise ValueError('budget_invalid:continuation_window')
                begin = _integer(window['pause_started_ms'], 'pause_started_ms')
                end = _integer(window['resumed_ms'], 'resumed_ms')
                increment = _vector(window['increment'], 'increment')
                extra_ms = _integer(window['elapsed_increment_ms'], 'elapsed_increment_ms')
                if (begin < previous_end or end < begin or end > obj.last_observed_at_ms
                        or end > _MAX_COUNTER or extra_ms > _MAX_COUNTER
                        or any(x > _MAX_COUNTER for x in increment.values())):
                    raise ValueError('budget_invalid:continuation_window')
                grants.add(window['grant_id'])
                excluded += end - begin
                for key in DIMENSIONS:
                    added[key] += increment[key]
                added_elapsed += extra_ms
                previous_end = end
            if (type(value['excluded_pause_ms']) is not int or value['excluded_pause_ms'] != excluded
                    or excluded > _MAX_COUNTER or obj.revision < len(rows) + len(windows)
                    or any(added[key] > obj.limits.total[key] or obj.limits.total[key] > _MAX_COUNTER for key in DIMENSIONS)
                    or added_elapsed > obj.limits.elapsed_ms or obj.limits.elapsed_ms > _MAX_COUNTER):
                raise ValueError('budget_invalid:continuation_window')
            obj._continuation_windows = deepcopy(windows)
            obj._excluded_pause_ms = excluded
        return obj

    @staticmethod
    def _validate_request(kind: str, amounts: dict[str, int]) -> None:
        if kind == "tool":
            if amounts != {"actions": 1, "model_calls": 0, "input_tokens": 0, "output_tokens": 0}:
                raise ValueError("budget_invalid:tool_reservation")
        elif kind == "model":
            if amounts["actions"] != 0 or amounts["model_calls"] != 1 or not all(amounts[k] > 0 for k in TOKEN_DIMENSIONS):
                raise ValueError("budget_invalid:model_reservation")
        else:
            raise ValueError("budget_invalid:kind")

    def usage(self) -> dict[str, Any]:
        used = {k: 0 for k in DIMENSIONS}
        unresolved = []
        unknown = []
        for key, row in self._reservations.items():
            if row["status"] == "released":
                continue
            values = row["actual"] if row["status"] == "settled" else row["reserved"]
            for dimension in DIMENSIONS:
                used[dimension] += values[dimension]
            if row["status"] == "reserved":
                unresolved.append(key)
            if row["unknown_tokens"]:
                unknown.append(key)
        return {"charged": used, "in_flight": unresolved, "unknown_usage": unknown}

    def within_work_limits(self) -> bool:
        """Check actual aggregate usage against work limits, retaining final reserve.

        Reservations are forecasts, not provider-enforced per-call caps. An
        overestimate or underestimate does not change the task-wide limits.
        Unknown usage is handled separately by the caller and reservation code.
        """
        charged = self.usage()['charged']
        return all(charged[key] <= self.limits.total[key] - self.limits.final_reserve[key]
                   for key in DIMENSIONS)

    def reserve(self, reservation_id: str, *, kind: str, amounts: Mapping[str, int],
                now_ms: int, lane: str = "work", cancelled: bool = False,
                _synthetic_allow_settled_unknown: bool = False) -> int:
        # Only the disposable synthetic evaluation store opts in; the full
        # reserved amount remains charged and unknown usage stays visible.
        if type(_synthetic_allow_settled_unknown) is not bool:
            raise ValueError("budget_invalid:synthetic_unknown_policy")
        if type(cancelled) is not bool:
            raise ValueError("budget_invalid:cancelled")
        if cancelled:
            raise BudgetExhausted("budget_task_cancelled")
        if not isinstance(reservation_id, str) or not reservation_id.strip():
            raise ValueError("budget_invalid:reservation_id")
        if reservation_id in self._reservations:
            raise ValueError("budget_duplicate_reservation")
        if lane not in {"work", "final"} or (lane == "final" and kind != "model"):
            raise ValueError("budget_invalid:lane")
        request = _vector(amounts, "amounts")
        self._validate_request(kind, request)
        now = _now(now_ms, self.last_observed_at_ms)

        def exhausted(reason: str) -> None:
            # Every valid observation of time must survive a rejected request,
            # not only a deadline rejection. Otherwise a clock rollback after
            # (for example) a call-count rejection can reopen the final window.
            if now > self.last_observed_at_ms:
                self.last_observed_at_ms = now
                self.revision += 1
            raise BudgetExhausted(reason)

        elapsed_limit = self.limits.elapsed_ms - (self.limits.final_reserve_ms if lane == "work" else 0)
        if now - self.started_at_ms - self._excluded_pause_ms >= elapsed_limit:
            exhausted("budget_elapsed_exhausted")
        if lane == "final" and any(r["lane"] == "final" and r["status"] != "released" for r in self._reservations.values()):
            exhausted("budget_final_attempt_used")
        usage = self.usage()
        if lane == "work" and any(r["lane"] == "final" and r["status"] != "released" for r in self._reservations.values()):
            exhausted("budget_final_already_started")
        if kind == "model" and ((usage["unknown_usage"] and not _synthetic_allow_settled_unknown) or any(
                r["kind"] == "model" and r["status"] == "reserved" for r in self._reservations.values())):
            exhausted("budget_model_usage_unresolved")
        for key in DIMENSIONS:
            limit = self.limits.total[key] - (self.limits.final_reserve[key] if lane == "work" else 0)
            if usage["charged"][key] + request[key] > limit:
                exhausted(f"budget_exhausted:{key}")
        self.last_observed_at_ms = now
        self._reservations[reservation_id] = {"lane": lane, "kind": kind, "reserved": request,
            "status": "reserved", "actual": None, "unknown_tokens": [], "outcome": None}
        self.revision += 1
        return self.revision

    def settle(self, reservation_id: str, *, outcome: str,
               input_tokens: int | None = None, output_tokens: int | None = None) -> int:
        row = self._reservations.get(reservation_id)
        if row is None or row["status"] != "reserved":
            raise ValueError("budget_reservation_not_pending")
        if outcome not in {"success", "failure", "timeout", "cancelled"}:
            raise ValueError("budget_invalid:outcome")
        actual = dict(row["reserved"])
        unknown: list[str] = []
        if row["kind"] == "model":
            for key, value in (("input_tokens", input_tokens), ("output_tokens", output_tokens)):
                if value is None:
                    unknown.append(key)  # Charge reservation, but do not pretend it was actual usage.
                else:
                    actual[key] = _integer(value, key)
        elif input_tokens is not None or output_tokens is not None:
            raise ValueError("budget_invalid:tool_token_usage")
        row.update(status="settled", actual=actual, unknown_tokens=unknown, outcome=outcome)
        self.revision += 1
        return self.revision

    def apply_continuation_window(self, *, grant_id: str, pause_started_ms: int, resumed_ms: int,
                                  increment: Mapping[str, int], increment_caps: Mapping[str, int],
                                  elapsed_increment_ms: int, elapsed_increment_cap_ms: int) -> int:
        """Apply one externally authenticated and bounded continuation grant.

        Experimental and unwired. The caller MUST authenticate and atomically
        consume a grant tied to this task/checkpoint, attest the exact quiescent
        pause interval, enforce cancellation/effect checks, and persist this
        revision before resuming. This method does none of those things. A grant
        ID can only be used once within this snapshot. Final-response use is
        terminal and cannot be reopened. All reservation records are preserved.
        """
        if not isinstance(grant_id, str) or not grant_id.strip() or len(grant_id) > 256:
            raise ValueError('budget_invalid:grant_id')
        if any(window['grant_id'] == grant_id for window in self._continuation_windows):
            raise ValueError('budget_continuation_grant_used')
        begin = _now(pause_started_ms, self.last_observed_at_ms)
        end = _now(resumed_ms, begin)
        increments = _vector(increment, 'increment')
        caps = _vector(increment_caps, 'increment_caps')
        elapsed = _integer(elapsed_increment_ms, 'elapsed_increment_ms')
        elapsed_cap = _integer(elapsed_increment_cap_ms, 'elapsed_increment_cap_ms')
        if (end > _MAX_COUNTER or elapsed > elapsed_cap or elapsed_cap > _MAX_COUNTER
                or any(increments[k] > caps[k] or caps[k] > _MAX_COUNTER
                       or self.limits.total[k] + increments[k] > _MAX_COUNTER for k in DIMENSIONS)
                or self.limits.elapsed_ms + elapsed > _MAX_COUNTER
                or self._excluded_pause_ms + end - begin > _MAX_COUNTER
                or (not any(increments.values()) and elapsed == 0)):
            raise ValueError('budget_invalid:continuation_bounds')
        usage = self.usage()
        if usage['in_flight'] or usage['unknown_usage']:
            raise ValueError('budget_continuation_usage_unresolved')
        if any(row['lane'] == 'final' and row['status'] != 'released' for row in self._reservations.values()):
            raise ValueError('budget_continuation_final_started')
        new_total = {k: self.limits.total[k] + increments[k] for k in DIMENSIONS}
        new_limits = BudgetLimits(new_total, self.limits.final_reserve,
                                  self.limits.elapsed_ms + elapsed, self.limits.final_reserve_ms)
        self._continuation_windows.append({'grant_id': grant_id, 'pause_started_ms': begin, 'resumed_ms': end,
                                           'increment': increments, 'elapsed_increment_ms': elapsed})
        self._excluded_pause_ms += end - begin
        self.last_observed_at_ms = end
        self.limits = new_limits
        self.revision += 1
        return self.revision

    def reconcile_unknown(self, reservation_id: str, *, input_tokens: int | None = None,
                          output_tokens: int | None = None) -> int:
        """Record subsequently obtained provider usage without changing the outcome.

        Only dimensions previously marked unknown may be reconciled. Evidence
        that usage is authoritative must be checked by the caller.
        """
        row = self._reservations.get(reservation_id)
        if row is None or row["status"] != "settled" or row["kind"] != "model":
            raise ValueError("budget_reservation_not_reconcilable")
        updates = {key: value for key, value in (("input_tokens", input_tokens),
                    ("output_tokens", output_tokens)) if value is not None}
        if not updates or any(key not in row["unknown_tokens"] for key in updates):
            raise ValueError("budget_invalid:reconciliation")
        normalized = {key: _integer(value, key) for key, value in updates.items()}
        row["actual"].update(normalized)
        row["unknown_tokens"] = [key for key in row["unknown_tokens"] if key not in normalized]
        self.revision += 1
        return self.revision

    def release_undispatched(self, reservation_id: str) -> int:
        """Release only with external proof no operation was dispatched.

        Never call this for an ambiguous crash, timeout, or uncertain effect.
        """
        row = self._reservations.get(reservation_id)
        if row is None or row["status"] != "reserved":
            raise ValueError("budget_reservation_not_pending")
        row["status"] = "released"
        self.revision += 1
        return self.revision


class BudgetRevisionConflict(ValueError):
    """A concurrent worker changed the persisted snapshot; reload before work."""


class SQLiteTaskBudgetStore:
    """Optional durable CAS store; no runtime is wired to it by this module.

    `reserve` atomically persists both successful reservations and rejected
    budget time observations. If using TaskBudget.reserve directly, a caller must
    CAS-save whenever revision changes, including on BudgetExhausted. `save`
    must succeed before dispatch. It never participates in the effect
    transaction: the caller remains responsible for operation idempotency and
    uncertain-effect reconciliation. Use a file-backed path for crash recovery.
    """

    def __init__(self, path: str) -> None:
        import sqlite3
        if not isinstance(path, str) or not path or path == ':memory:':
            raise ValueError('budget_store_file_required')
        self.path = path
        with sqlite3.connect(self.path) as conn:
            conn.execute('''CREATE TABLE IF NOT EXISTS task_budget_snapshots (
                task_id TEXT PRIMARY KEY, revision INTEGER NOT NULL,
                snapshot_json TEXT NOT NULL
            ) STRICT''')

    @staticmethod
    def _task_id(task_id: str) -> str:
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError('budget_invalid:task_id')
        return task_id

    def create(self, task_id: str, budget: TaskBudget) -> None:
        import json
        import sqlite3
        self._task_id(task_id)
        if not isinstance(budget, TaskBudget) or budget.revision != 0:
            raise ValueError('budget_store_initial_state_required')
        try:
            with sqlite3.connect(self.path) as conn:
                conn.execute('INSERT INTO task_budget_snapshots VALUES (?, ?, ?)',
                             (task_id, budget.revision, json.dumps(budget.to_dict(), sort_keys=True)))
        except sqlite3.IntegrityError as exc:
            raise BudgetRevisionConflict('budget_already_exists') from exc

    def load(self, task_id: str) -> TaskBudget:
        import json
        import sqlite3
        self._task_id(task_id)
        with sqlite3.connect(self.path) as conn:
            row = conn.execute('SELECT revision, snapshot_json FROM task_budget_snapshots WHERE task_id=?', (task_id,)).fetchone()
        if row is None:
            raise KeyError(task_id)
        try:
            budget = TaskBudget.from_dict(json.loads(row[1]))
        except (ValueError, TypeError, KeyError) as exc:
            raise ValueError('budget_store_snapshot_invalid') from exc
        if budget.revision != row[0]:
            raise ValueError('budget_store_revision_mismatch')
        return budget

    def save(self, task_id: str, budget: TaskBudget, *, expected_revision: int) -> None:
        import json
        import sqlite3
        self._task_id(task_id)
        _integer(expected_revision, 'expected_revision')
        if not isinstance(budget, TaskBudget) or budget.revision != expected_revision + 1:
            raise ValueError('budget_store_single_transition_required')
        with sqlite3.connect(self.path) as conn:
            cursor = conn.execute('UPDATE task_budget_snapshots SET revision=?, snapshot_json=? WHERE task_id=? AND revision=?',
                (budget.revision, json.dumps(budget.to_dict(), sort_keys=True), task_id, expected_revision))
            if cursor.rowcount != 1:
                raise BudgetRevisionConflict('budget_revision_conflict')

    def reserve(self, task_id: str, *, expected_revision: int, **kwargs: Any) -> TaskBudget:
        """Fence and persist a reservation OR a rejected time observation.

        On concurrent modification this raises BudgetRevisionConflict; callers
        must reload. No model/tool may be dispatched if this call raises.
        """
        if '_synthetic_allow_settled_unknown' in kwargs and type(self) is not SyntheticEvaluationBudgetStore:
            raise ValueError('synthetic_budget_policy_not_available')
        budget = self.load(task_id)
        if budget.revision != expected_revision:
            raise BudgetRevisionConflict('budget_revision_conflict')
        try:
            budget.reserve(**kwargs)
        except BudgetExhausted:
            if budget.revision != expected_revision:
                self.save(task_id, budget, expected_revision=expected_revision)
            raise
        self.save(task_id, budget, expected_revision=expected_revision)
        return budget


class SyntheticEvaluationBudgetStore(SQLiteTaskBudgetStore):
    """Disposable comparison only: retain and fully charge unreported usage.

    Not a billing estimate, production budget, or crash-reconciliation policy.
    A previous *in-flight* model call still blocks; each settled unknown keeps
    its full reservation charged and is explicitly recorded as unknown.
    """

    def reserve(self, task_id: str, *, expected_revision: int, **kwargs: Any) -> TaskBudget:
        if '_synthetic_allow_settled_unknown' in kwargs:
            raise ValueError('synthetic_budget_policy_not_overridable')
        return super().reserve(task_id, expected_revision=expected_revision,
                               _synthetic_allow_settled_unknown=True, **kwargs)
