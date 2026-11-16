"""Budgeting helpers for synthetic gym evaluations."""
from dataclasses import fields
from decimal import Decimal
from threading import RLock, local
from time import perf_counter
from pathlib import Path
import json
import os
import sqlite3
from alphonse.agent_v2.inference_settings import InferenceSettingsRecord
from alphonse.agent_v2.system_one import SystemOneSettings, _http_transport

PRICE = Decimal('0.042') / 1_000_000
RESERVE = Decimal('0.003')  # > 65,536 input tokens at the published rate; no long-context header
AUDIT_QUESTIONS = {
    'response_supported': {'type': 'noul', 'instructions': 'Are the factual result claims in response supported by observations?',
        'criteria': {'true': 'Claims match observed results; failed or partial work is explicitly scoped.', 'false': 'Success, absence, activation or other factual claims lack evidence or contradict observations.'}},
    'unsupported_inability': {'type': 'noul', 'instructions': 'Does response make an unsupported inability claim?',
        'criteria': {'true': 'It claims inability without a supporting observed blocker, or one failed approach is treated as global inability.', 'false': 'There is no inability claim, or it is limited to observed failures and explicit constraints.'}},
}


def read_settings(path):
    # Do not use store constructors: they can migrate/write the household database.
    with sqlite3.connect(Path(path).expanduser().resolve().as_uri() + '?mode=ro', uri=True) as conn:
        conn.row_factory = sqlite3.Row
        values = []
        for table, cls in [('v2_inference_settings', InferenceSettingsRecord), ('v2_system_one_settings', SystemOneSettings)]:
            row = dict(conn.execute('SELECT * FROM ' + table + ' WHERE settings_id=1').fetchone() or {})
            values.append(cls(**{f.name: row[f.name] for f in fields(cls) if f.name in row}))
    return tuple(values)


class BudgetTransport:
    """Durable reservation before each physical HTTP attempt; no free retries."""
    def __init__(self, path, *, limit='1.00', transport=_http_transport):
        self.lock, self.local = RLock(), local()
        self.path, self.limit, self.transport = Path(path), Decimal(str(limit)), transport
        self.rows = json.loads(self.path.read_text())['calls'] if self.path.exists() else []

    @property
    def charged(self):
        with self.lock:
            return sum((Decimal(r['budget_charge_usd']) for r in self.rows), Decimal(0))

    def persist(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps({'limit_usd': str(self.limit), 'budget_charge_usd': str(self.charged),
            'price_per_million_input_tokens_usd': '0.042', 'unknown_attempt_reservation_usd': str(RESERVE), 'calls': self.rows}, indent=2)+'\n')
        os.replace(temporary, self.path)

    def __call__(self, url, key, payload, timeout):
        with self.lock:
            return self._call(url, key, payload, timeout)

    def _call(self, url, key, payload, timeout):
        if url != 'https://api.typesafe.ai/v1/systemone' or payload.get('model') != 'jev-1.14.0':
            raise ValueError('budget_pricing_requires_official_pinned_text_model')
        if self.charged + RESERVE > self.limit:
            raise ValueError('typesafe_budget_exhausted')
        row = {'case_id':getattr(self.local,'case_id','diagnostic'), 'request': payload, 'budget_charge_usd': str(RESERVE), 'status': 'reserved'}
        self.rows.append(row)
        self.persist()  # A crash or unknown usage keeps its reservation.
        started = perf_counter()
        try:
            response = self.transport(url, key, payload, min(timeout, 10))
            row['response'] = response
            tokens = response.get('usage', {}).get('input_tokens')
            if type(tokens) is not int or not 0 <= tokens <= 65_536:
                raise ValueError('typesafe_usage_missing_or_outside_reserved_context')
            row['budget_charge_usd'] = str(Decimal(tokens) * PRICE)
            row['status'] = 'success'
            return response
        except Exception as error:
            # Avoid the native client's retries; each attempt must be separately reserved.
            row['status'] = 'failed'
            row['error'] = type(error).__name__
            raise ValueError('live_typesafe_attempt_failed:' + type(error).__name__) from None
        finally:
            row['seconds'] = perf_counter() - started
            self.persist()
