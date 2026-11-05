"""Offline synthetic lexical recall coverage; no answer quality or rollout claims."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.tools.registry.native.memory import execute_search_memory


def run(corpus: dict) -> dict:
    if corpus.get('schema_version') != 1 or corpus.get('synthetic') is not True:
        raise ValueError('synthetic_recall_corpus_required')
    rows = []
    for case in corpus['cases']:
        store = SQLiteConversationStore(':memory:')
        try:
            source = store.record(owner_user_id='member-a', content=case['text'], role='user',
                                  source='synthetic', source_message_id='target',
                                  created_at=case.get('created_at', '2026-10-17T12:00:00+00:00'))
            store.record(owner_user_id='member-b', content='PRIVATE ' + case['text'], role='user',
                         source='synthetic', source_message_id='private')
            context = SimpleNamespace(task=TaskState(user='member-a'), memory=None, conversation_store=store,
                                      user_timezone_provider=lambda _: case.get('timezone', 'America/Mexico_City'))
            arguments = {'query': case['query'], 'source_types': ['conversation']}
            arguments.update({key: case[key] for key in ('start_date', 'end_date') if key in case})
            result = execute_search_memory(arguments, context=context)
            retrieved = ['target' for hit in result['results'] if hit['source_id'] == f'conversation:{source.event_id}']
            expected = set(case['expected'])
            rows.append({'id': case['id'], 'category': case['category'], 'expected': sorted(expected),
                         'retrieved': retrieved, 'exact_match': set(retrieved) == expected,
                         'necessary_source_recall': len(set(retrieved) & expected) / len(expected) if expected else None,
                         'private_leak': 'PRIVATE' in json.dumps(result), 'status': result['status']})
        finally:
            store._memory.close()
    categories = {}
    for category in sorted({row['category'] for row in rows}):
        group = [row for row in rows if row['category'] == category]
        categories[category] = {'cases': len(group), 'exact_matches': sum(row['exact_match'] for row in group),
                                'exact_match_rate': sum(row['exact_match'] for row in group) / len(group)}
    return {'schema_version': 1, 'synthetic': True, 'backend': 'sqlite_fts5', 'model_calls': 0,
            'corpus_sha256': hashlib.sha256(json.dumps(corpus, sort_keys=True).encode()).hexdigest(),
            'categories': categories, 'cases': rows,
            'limitations': ['Small synthetic retrieval coverage only; not answer quality or a rollout gate.',
                            'Lexical retrieval has no translation or semantic synonym guarantee.',
                            'No production scale latency measurement; authorized in-memory index is rebuilt per call.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = run(json.loads(args.corpus.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
