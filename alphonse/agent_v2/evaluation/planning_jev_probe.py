"""Three budgeted Jev diagnostics over disposable synthetic fragment data."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from alphonse.agent_v2.database import default_database_path
from alphonse.agent_v2.evaluation.resourceful_live import BudgetTransport, read_settings
from alphonse.agent_v2.system_one import JevCriterionDecisionProvider


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ledger', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--settings-db', type=Path, default=default_database_path())
    args = parser.parse_args()
    _, settings = read_settings(args.settings_db)
    provider_settings = replace(settings, model='jev-1.14.0')
    budget = BudgetTransport(args.ledger, limit='0.02')
    provider = JevCriterionDecisionProvider(provider_settings, transport=budget)
    phase = {'phase_id': 'synthetic', 'objective': 'Read four requested fragments', 'subgoals': []}
    subgoal = {'subgoal_id': 'read', 'objective': 'Read all four fragments',
               'required_output_type': 'fragments', 'completion': {'kind': 'output_present'}}
    cases = [
        ('partial_useful', {'fragment': 'north'}, []),
        ('empty_result', {'fragments': []}, [{'status': 'success', 'result': {'fragments': []}}]),
        ('all_fragments', {'fragments': ['north', 'east', 'south', 'west']}, []),
    ]
    results = []
    for name, result, earlier in cases:
        budget.local.case_id = name
        action = {'action_id': 'synthetic-action', 'subgoal_id': 'read', 'tool_id': 'native.fixture_read',
                  'arguments': {}, 'status': 'success', 'result': result}
        current = {'evidence_ref': 'fixture:current', 'phase_id': 'synthetic', 'subgoal_id': 'read',
                   'status': 'success', 'result': result}
        try:
            review = provider.evaluate_tactical_progress(goal='Read four synthetic fragments', phase=phase,
                subgoal=subgoal, action=action, execution_log=earlier + [current])
            results.append({'case': name, 'answers': review.answers, 'complete': review.complete,
                            'context_complete': review.context_complete, 'duration_ms': review.duration_ms,
                            'usage': review.usage})
        except Exception as exc:
            results.append({'case': name, 'error_type': type(exc).__name__})
    report = {'results': results, 'budget_charge_usd': str(budget.charged), 'limit_usd': '0.02'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
    return 1 if any('error_type' in item for item in results) else 0


if __name__ == '__main__':
    raise SystemExit(main())
