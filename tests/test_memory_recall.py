"""Structured retrieval, authorization, pagination and original-source recovery."""
import json
from types import SimpleNamespace

import pytest

from alphonse.agent_v2.conversations import SQLiteConversationStore
from alphonse.agent_v2.core.intelligence.task_state import TaskState
from alphonse.agent_v2.core.memory.contracts import conversation_item
from alphonse.agent_v2.core.memory.knowledge import KnowledgeMemory, KnowledgeStore
from alphonse.agent_v2.core.memory.ledger import LedgerMemory
from alphonse.agent_v2.memory_settings import SQLiteMemorySettingsStore
from alphonse.agent_v2.core.tools.registry.native.memory import (
    execute_search_memory as search, execute_read_memory_source as read,
    build_read_memory_source_tool_definition, build_search_memory_tool_definition,
)


@pytest.fixture
def context(tmp_path):
    conversations = SQLiteConversationStore(tmp_path / 'memory.sqlite')
    knowledge = KnowledgeMemory(store=KnowledgeStore(tmp_path / 'memory.sqlite'), conversations=conversations,
                               extractor=lambda *a, **kw: {'candidates': []}, reviewer=lambda *a: [])
    ledger = LedgerMemory(users_root=lambda: tmp_path, settings_store=SQLiteMemorySettingsStore(), knowledge=knowledge)
    return SimpleNamespace(task=TaskState(user='alex'), memory=ledger, conversation_store=conversations,
                           user_timezone_provider=lambda user: 'America/Mexico_City')


def event(context, text='Receipt ABC-12', **kw):
    values = dict(owner_user_id='alex', content=text, source='test', role='user',
                  source_message_id=str(len(context.conversation_store.list_all(owner_user_id='alex'))) + text,
                  created_at='2026-10-17T05:59:00+00:00')
    values.update(kw)
    return context.conversation_store.record(**values)


def test_search_all_types_and_expand_originals(context):
    source = event(context)
    outputs = [{'content': source.content, 'kind': 'fact', 'subjects': ['receipt'],
                'claim_key': 'receipt.id', 'quote': source.content}]
    knowledge = context.memory.knowledge
    knowledge.extractor = lambda *a, **kw: {'candidates': outputs}
    knowledge.reviewer = lambda payload, candidates: [True] * len(candidates)
    knowledge.process_event(source, 'alex')
    knowledge.store.capture_task(TaskState(user='alex', task_id='receipt-task', evidence_journal=[
        {'status': 'failed', 'result': {'error': 'ABC-12 timeout'}}]))
    root = context.memory._project_memory_dir('alex', '')
    (root / 'legacy').mkdir()
    (root / 'legacy/ledger-1.md').write_text('Original ABC-12 legacy task audit')
    result = search({'query': 'ABC-12'}, context=context)
    assert result['status'] == 'found'
    assert {r['source_type'] for r in result['results']} == {'conversation', 'knowledge', 'task_evidence', 'legacy_ledger'}
    for hit in result['results']:
        expanded = read({'source_id': hit['source_id']}, context=context)
        assert expanded['status'] == 'found'
        assert 'ABC-12' in expanded['items'][0]['content']
        if hit['source_type'] == 'knowledge':
            assert expanded['supporting_sources'][0]['source_id'] == f'conversation:{source.event_id}'
        if hit['source_type'] == 'task_evidence':
            assert 'failed' in expanded['items'][0]['content']
    assert len({h['source_id'] for h in result['results']}) == 4
    memory_id = next(h['source_id'] for h in result['results'] if h['source_type'] == 'knowledge')
    assert context.conversation_store.delete(event_id=source.event_id, owner_user_id='alex')
    assert read({'source_id': f'conversation:{source.event_id}'}, context=context)['status'] == 'not_found'
    assert read({'source_id': memory_id}, context=context)['status'] == 'not_found'
    context.task.user = 'other-member'
    for hit in result['results']:
        assert read({'source_id': hit['source_id']}, context=context)['status'] == 'not_found'


def test_visibility_before_rank_and_live_expansion(context):
    public = event(context, 'Shared receipt', owner_user_id='sam', visibility='shared', shared_with=('alex',))
    private = event(context, 'PRIVATE receipt', owner_user_id='sam')
    public_id = f'conversation:{public.event_id}'
    result = search({'query': 'receipt', 'source_types': ['conversation']}, context=context)
    assert [r['source_id'] for r in result['results']] == [public_id]
    score = result['results'][0]['score']
    for i in range(10):
        event(context, 'PRIVATE receipt ' + str(i), owner_user_id='sam')
    assert search({'query': 'receipt', 'source_types': ['conversation']}, context=context)['results'][0]['score'] == score
    assert read({'source_id': f'conversation:{private.event_id}'}, context=context) == read({'source_id': 'conversation:missing'}, context=context)
    with context.conversation_store._connect() as conn:
        conn.execute("UPDATE v2_conversation_events SET visibility='private', shared_with='[]' WHERE event_id=?", (public.event_id,))
    assert read({'source_id': public_id}, context=context)['status'] == 'not_found'


def test_neighbors_and_long_quote_pagination(context):
    before = event(context, 'Question before', project_id='old')
    text = 'Exact quoted passage\n' + ('original text ' * 800).rstrip()
    anchor = event(context, text, project_id='new', role='assistant')
    event(context, 'PRIVATE NEIGHBOR', owner_user_id='sam')
    after = event(context, 'Follow-up after')
    source_id = f'conversation:{anchor.event_id}'
    hit = search({'query': 'quoted', 'source_types': ['conversation']}, context=context)['results'][0]
    assert hit['truncated'] and hit['total_chars'] == len(text)
    chunks, offset = [], 0
    while True:
        result = read({'source_id': source_id, 'offset': offset, 'max_chars': 1024}, context=context)
        chunks.append(result['items'][0]['content'])
        assert {n['source']['source_id'] for n in result['neighbors']} == {f'conversation:{before.event_id}', f'conversation:{after.event_id}'}
        if result['next_offset'] is None:
            break
        offset = result['next_offset']
    assert ''.join(chunks) == text
    assert 'PRIVATE NEIGHBOR' not in json.dumps(result)


def test_timezone_dates_entities_and_paging(context):
    yesterday = event(context, 'ABC-12 José receipt before midnight')
    today = event(context, 'ABC-12 José receipt after midnight', created_at='2026-10-17T06:00:00+00:00')
    args = {'query': 'ABC-12', 'source_types': ['conversation'], 'entities': ['Jose'], 'limit': 1}
    first = search(args, context=context)
    second = search({**args, 'offset': first['next_offset']}, context=context)
    assert first['snapshot'] == second['snapshot']
    assert first['results'][0]['source_id'] != second['results'][0]['source_id']
    assert second['next_offset'] is None
    dated = search({**args, 'start_date': '2026-10-17', 'end_date': '2026-10-17'}, context=context)
    assert dated['results'][0]['source_id'] == f'conversation:{today.event_id}'
    assert dated['results'][0]['event_time'] == today.observed_at
    assert yesterday.observed_at == '2026-10-17T05:59:00+00:00'
    assert search({**args, 'start_date': '2026-10-18'}, context=context)['status'] == 'empty'


def test_statuses_and_backend_exception_redaction(context, monkeypatch):
    assert search({'query': 'missing'}, context=context)['status'] == 'empty'
    def fail(*args, **kw):
        raise RuntimeError('PRIVATE exception /path')
    context.memory.knowledge.recall = fail
    result = search({'query': 'missing'}, context=context)
    assert result['status'] == 'partial' and result['coverage']['knowledge'] == 'failed'
    assert 'PRIVATE' not in json.dumps(result)
    event(context)
    assert search({'query': 'receipt'}, context=context)['status'] == 'partial'
    context.conversation_store.recall_items = fail
    assert search({'query': 'receipt', 'source_types': ['conversation']}, context=context)['status'] == 'failed'
    context.conversation_store = None
    assert search({'query': 'receipt', 'source_types': ['conversation']}, context=context)['status'] == 'unavailable'
    import alphonse.agent_v2.core.tools.registry.native.memory as native
    monkeypatch.setattr(native, 'rank', fail)
    assert search({'query': 'receipt'}, context=context)['status'] == 'failed'


def test_adversarial_snapshot_and_source_reader_fail_closed(context):
    hidden = event(context, 'PRIVATE receipt', owner_user_id='sam')
    shared = event(context, 'Shared receipt', owner_user_id='sam', visibility='shared', shared_with=('alex',))
    from alphonse.agent_v2.core.memory.contracts import ContextItem, SourceReference, AccessScope
    leaked = ContextItem(SourceReference('conversation:hidden', 'conversation', AccessScope('sam')), hidden.content)
    visible = conversation_item(shared, 'alex')
    context.conversation_store.recall_items = lambda requester: [leaked, visible, visible]
    result = search({'query': 'receipt', 'source_types': ['conversation']}, context=context)
    assert len(result['results']) == 1 and 'PRIVATE' not in json.dumps(result)
    context.conversation_store.read = lambda **kw: hidden
    assert read({'source_id': f'conversation:{hidden.event_id}'}, context=context)['status'] == 'not_found'


def test_legacy_symlink_and_path_id_cannot_escape(context, tmp_path):
    root = context.memory._project_memory_dir('alex', '')
    (root / 'legacy').mkdir()
    secret = tmp_path / 'outside.md'
    secret.write_text('SECRET receipt')
    (root / 'legacy/ledger-escape.md').symlink_to(secret)
    assert search({'query': 'receipt'}, context=context)['status'] == 'empty'
    assert read({'source_id': str(secret)}, context=context)['status'] == 'not_found'
    assert read({'source_id': 'legacy-ledger:../../outside.md'}, context=context)['status'] == 'not_found'


@pytest.mark.parametrize('args', [
    {'query': ''}, {'query': 'x' * 513}, {'query': 'receipt', 'limit': True},
    {'query': 'receipt', 'limit': 31}, {'query': 'receipt', 'offset': -1},
    {'query': 'receipt', 'source_types': ['unknown']}, {'query': 'receipt', 'source_types': [{}]},
    {'query': 'receipt', 'entities': ['']}, {'query': 'receipt', 'historical': 'yes'},
    {'query': 'receipt', 'start_date': 'bad'},
    {'query': 'receipt', 'start_date': '2026-10-20', 'end_date': '2026-10-17'},
])
def test_bounded_inputs(context, args):
    with pytest.raises(ValueError):
        search(args, context=context)


def test_tool_schemas_registered_and_read_only():
    for definition in (build_search_memory_tool_definition(), build_read_memory_source_tool_definition()):
        assert definition.descriptor.read_only
        assert definition.argument_schema['additionalProperties'] is False


def test_synthetic_coverage_report_is_reproducible_without_network(monkeypatch):
    from pathlib import Path
    import socket
    from alphonse.agent_v2.evaluation.memory_recall import run
    def no_network(*args, **kwargs):
        raise AssertionError('recall evaluation attempted network')
    monkeypatch.setattr(socket, 'socket', no_network)
    root = Path(__file__).parents[1]
    corpus = json.loads((root / 'tests/fixtures/memory_recall/v1.json').read_text())
    report = run(corpus)
    assert report == run(corpus) == json.loads((root / 'docs/v3-hierarchical-capd/evidence/memory-recall-v1/results.json').read_text())
    assert all(not case['private_leak'] for case in report['cases'])
    assert report['categories']['identifier']['exact_match_rate'] == 1
    assert report['categories']['temporal']['exact_match_rate'] == 1
    assert report['categories']['multilingual']['exact_matches'] == 2
    assert report['categories']['paraphrase']['exact_matches'] == 1


def test_source_substitution_and_unbounded_alternate_neighbors_are_rejected(context):
    anchor = event(context, 'Original anchor')
    other = event(context, 'Different visible source')
    source_id = f'conversation:{anchor.event_id}'
    original_read = context.conversation_store.read
    context.conversation_store.read = lambda **kw: other
    assert read({'source_id': source_id}, context=context)['status'] == 'not_found'
    context.conversation_store.read = original_read
    context.conversation_store.read_neighbors = lambda **kw: [conversation_item(other, 'alex')] * 100
    assert len(read({'source_id': source_id, 'neighbors': 1}, context=context)['neighbors']) <= 2
    assert read({'source_id': source_id, 'neighbors': 0}, context=context)['neighbors'] == []
