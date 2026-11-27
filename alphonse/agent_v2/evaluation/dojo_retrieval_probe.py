"""Reproduce search behavior on the disposable Japanese Dojo conversation.

This is a diagnostic of today's retrieval implementation, not a model or
language-reliability test. It opens an in-memory synthetic owner store only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.memory.recall import keywords
from alphonse.agent_v2.core.tools.registry.native.memory import execute_search_memory
from alphonse.agent_v2.evaluation.dojo_conversation_reliability import CASES, FILLER


def run() -> dict:
    case = CASES['japanese-owner-change']
    history = (*case.older, *FILLER, *case.recent)
    queries = ('家族映画 週末 決めた', '週末 家族映画 映画 決めた',
               '家族映画', '月の庭', '家族映画は')
    store = SQLiteConversationStore(':memory:')
    try:
        for index, (role, content) in enumerate(history):
            store.record(owner_user_id='synthetic-owner', role=role, content=content,
                         source='synthetic', source_message_id=f'synthetic:{index}')
        context = SimpleNamespace(task=TaskState(user='synthetic-owner'), memory=None,
                                  conversation_store=store, user_timezone_provider=lambda _: 'UTC')
        rows = []
        for query in queries:
            result = execute_search_memory({'query': query, 'source_types': ['conversation'], 'limit': 30},
                                           context=context)
            rows.append({'query': query, 'terms': keywords(query), 'status': result['status'],
                         'coverage': result['coverage'], 'backend': result['backend'],
                         'total_matches': result['total_matches'], 'next_offset': result['next_offset'],
                         'later_owner_choice_returned': any(case.markers[0] in row['excerpt']
                                                            and row['source'].get('role') == 'user'
                                                            for row in result['results']),
                         'excerpts': sorted(row['excerpt'] for row in result['results'])})
        with sqlite3.connect(':memory:') as conn:
            conn.execute("CREATE VIRTUAL TABLE docs USING fts5(content, tokenize='unicode61')")
            conn.execute('CREATE VIRTUAL TABLE vocab USING fts5vocab(docs, row)')
            conn.executemany('INSERT INTO docs(content) VALUES (?)', [(content,) for _, content in case.older])
            terms = [row[0] for row in conn.execute('SELECT term FROM vocab ORDER BY term')]
            direct = []
            for query in queries:
                match = ' OR '.join('"' + term + '"' for term in keywords(query))
                count = conn.execute('SELECT count(*) FROM docs WHERE docs MATCH ?', (match,)).fetchone()[0]
                direct.append({'query': query, 'match': match, 'older_records_matched': count})
        return {'fixture': case.case_id, 'synthetic': True, 'search_results': rows,
                'direct_fts_indexed_terms': terms, 'direct_fts_results': direct,
                'limitations': ['Isolated reproduction, not the original live tool observations.',
                    'Direct FTS probe indexes only the older records; the tool probe uses the full synthetic history.',
                    'Results do not assess relevance on a broader corpus, model behavior, or general language reliability.',
                    'Complete snapshot coverage does not establish semantic recall.']}
    finally:
        store._memory.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', type=Path)
    args = parser.parse_args()
    with args.journal.open('x', encoding='utf-8') as journal:
        json.dump(run(), journal, ensure_ascii=False, sort_keys=True, indent=2)
        journal.write('\n')


if __name__ == '__main__':
    main()
