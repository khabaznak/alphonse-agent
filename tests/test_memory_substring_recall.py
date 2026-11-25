"""Bounded literal fallback over authorized recall snapshots."""
import json
from types import SimpleNamespace

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.tools.registry.native.memory import execute_search_memory


def test_multiterm_substrings_are_scoped_and_paginate_stably(tmp_path):
    store = SQLiteConversationStore(tmp_path / 'recall.sqlite')
    owner = 'synthetic-owner'
    visible = []
    for i in range(95):
        visible.append(store.record(owner_user_id=owner, role='user', source='fixture',
            content=f'週末の家族映画は作品番号{i}です。', source_message_id=f'visible-{i}'))
    private = store.record(owner_user_id='another-owner', role='user', source='fixture',
        content='週末の家族映画は秘密の作品です。', source_message_id='private')
    context = SimpleNamespace(task=TaskState(user=owner), memory=None, conversation_store=store)
    args = {'query': '家族映画 週末 決めた', 'source_types': ['conversation'], 'limit': 30}
    pages = []
    offset = 0
    while offset is not None:
        page = execute_search_memory({**args, 'offset': offset}, context=context)
        pages.append(page)
        offset = page['next_offset']
    ids = [row['source_id'] for page in pages for row in page['results']]
    assert len(pages) == 4 and all(page['status'] == 'found' for page in pages)
    assert all(page['total_matches'] == 95 for page in pages)
    assert len({page['snapshot'] for page in pages}) == 1
    assert set(ids) == {f'conversation:{record.event_id}' for record in visible}
    assert private.event_id not in json.dumps(pages)
    assert '秘密' not in json.dumps(pages, ensure_ascii=False)


def test_short_terms_do_not_expand_and_fts_hits_precede_substrings(tmp_path):
    store = SQLiteConversationStore(tmp_path / 'recall.sqlite')
    def add(text, sid):
        return store.record(owner_user_id='owner', role='user', source='fixture', content=text,
                            source_message_id=sid)
    exact = add('A catalog of plants.', 'exact')
    partial = add('A cataloguing reference.', 'partial')
    add('an embedded abcd token', 'short')
    context = SimpleNamespace(task=TaskState(user='owner'), memory=None, conversation_store=store)
    result = execute_search_memory({'query': 'catalog unknown', 'source_types': ['conversation']}, context=context)
    assert [row['source_id'] for row in result['results']] == [
        f'conversation:{exact.event_id}', f'conversation:{partial.event_id}']
    assert execute_search_memory({'query': 'ab zx', 'source_types': ['conversation']}, context=context)['status'] == 'empty'


def test_substring_uses_unicode_normalization_and_bounded_query_terms(tmp_path):
    store = SQLiteConversationStore(tmp_path / 'recall.sqlite')
    record = store.record(owner_user_id='owner', role='user', source='fixture',
        content='Une dégustation de caféine et une promenade.', source_message_id='accent')
    context = SimpleNamespace(task=TaskState(user='owner'), memory=None, conversation_store=store)
    result = execute_search_memory({'query': 'CAFE impossible', 'source_types': ['conversation']}, context=context)
    assert result['results'][0]['source_id'] == f'conversation:{record.event_id}'
    # Only the first 24 deduplicated search terms are eligible.
    terms = [f'word{i}' for i in range(24)] + ['cafe']
    assert execute_search_memory({'query': ' '.join(terms), 'source_types': ['conversation']}, context=context)['status'] == 'empty'
