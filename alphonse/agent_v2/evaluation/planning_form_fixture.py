"""Disposable in-memory two-field form for orchestration comparisons.

This is test infrastructure, not a browser backend or an authorization system.
The fixture independently records effects and rejects stale, unauthorized or
unsafe operations so comparisons can check actual outcomes.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any


class FixtureDenied(ValueError):
    pass


@dataclass
class FormFixture:
    fields: dict[str, str] = field(default_factory=lambda: {'name': 'Synthetic Robin', 'code': 'fixture-17'})
    actor: str = 'fixture-owner'
    allowed: bool = True
    version: int = 1
    state: dict[str, str] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    submitted: dict[str, str] | None = None
    unconfirmed: bool = False
    cancelled: bool = False

    def _check(self, actor: str, *, write: bool = False, version: int | None = None) -> None:
        if self.cancelled:
            raise FixtureDenied('cancelled')
        if actor != self.actor or not self.allowed:
            raise FixtureDenied('access_denied')
        if write and version != self.version:
            raise FixtureDenied('stale_form')
        if write and (self.submitted is not None or self.unconfirmed):
            raise FixtureDenied('effect_requires_reconciliation')

    def inspect(self, actor: str) -> dict[str, Any]:
        self._check(actor)
        self.events.append({'operation': 'inspect', 'version': self.version})
        return {'version': self.version, 'fields': list(self.fields), 'values': dict(self.state)}

    def fill(self, actor: str, version: int, field_name: str, value: str) -> dict[str, Any]:
        self._check(actor, write=True, version=version)
        if field_name not in self.fields:
            raise FixtureDenied('unknown_field')
        self.state[field_name] = value
        self.events.append({'operation': 'fill', 'field': field_name, 'version': version})
        return {'field': field_name, 'version': version}

    def submit(self, actor: str, version: int, *, ambiguous: bool = False) -> dict[str, Any]:
        self._check(actor, write=True, version=version)
        if self.state != self.fields:
            raise FixtureDenied('incomplete_or_wrong_values')
        self.submitted = dict(self.state)
        self.unconfirmed = ambiguous
        self.events.append({'operation': 'submit', 'version': version})
        return {'status': 'uncertain' if ambiguous else 'accepted'}

    def verify(self, actor: str) -> dict[str, Any]:
        self._check(actor)
        self.events.append({'operation': 'verify', 'version': self.version})
        self.unconfirmed = False
        return {'receipt_id': 'fixture-receipt-1' if self.submitted is not None else None,
                'submitted': deepcopy(self.submitted), 'version': self.version}

    def change_form(self, fields: dict[str, str]) -> None:
        self.fields = dict(fields)
        self.state.clear()
        self.version += 1
        self.events.append({'operation': 'change', 'version': self.version})

    def snapshot(self) -> dict[str, Any]:
        return deepcopy(self.__dict__)

    @classmethod
    def restore(cls, data: dict[str, Any]) -> 'FormFixture':
        return cls(**deepcopy(data))

    def oracle(self) -> dict[str, Any]:
        submits = [i for i, event in enumerate(self.events) if event['operation'] == 'submit']
        verified = bool(submits and any(e['operation'] == 'verify' for e in self.events[submits[-1] + 1:]))
        return {'verified': self.submitted == self.fields and verified,
                'submission_count': len(submits), 'expected_values': self.submitted == self.fields}
