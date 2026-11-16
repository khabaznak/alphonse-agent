from decimal import Decimal

import pytest

from alphonse.agent_v2.evaluation.gym_budget import BudgetTransport, RESERVE

PAYLOAD = {'model':'jev-1.14.0','state':{'synthetic':True},'questions':{'q':{'type':'noul','instructions':'Is this synthetic?'}}}
URL = 'https://api.typesafe.ai/v1/systemone'


def test_budget_reserves_before_network_and_accounts_reported_usage(tmp_path):
    path = tmp_path / 'ledger.json'
    def send(url, key, payload, timeout):
        assert path.exists()
        assert BudgetTransport(path).charged == RESERVE
        return {'usage':{'input_tokens':100},'answers':{'q':{'type':'noul','noul':.99}}}
    budget=BudgetTransport(path,transport=send)
    budget(URL,'DO-NOT-STORE',PAYLOAD,10)
    assert budget.charged == Decimal('0.0000042')
    assert 'DO-NOT-STORE' not in path.read_text()


def test_uncertain_network_cost_survives_restart_and_blocks_overspending(tmp_path):
    path=tmp_path/'ledger.json'
    calls=[]
    def fail(*args):
        calls.append(True)
        raise TimeoutError()
    budget=BudgetTransport(path,limit='0.003',transport=fail)
    with pytest.raises(ValueError,match='attempt_failed'):
        budget(URL,'synthetic',PAYLOAD,10)
    restored=BudgetTransport(path,limit='0.003',transport=fail)
    assert restored.charged == RESERVE
    with pytest.raises(ValueError,match='budget_exhausted'):
        restored(URL,'synthetic',PAYLOAD,10)
    assert len(calls)==1


def test_unknown_endpoint_or_model_cannot_use_verified_price(tmp_path):
    budget=BudgetTransport(tmp_path/'ledger.json',transport=lambda *_:pytest.fail('must not send'))
    for url,payload in [('https://example.com',PAYLOAD),(URL,dict(PAYLOAD,model='other-model'))]:
        with pytest.raises(ValueError,match='official_pinned_text_model'):
            budget(url,'synthetic',payload,10)
    assert budget.charged==0


def test_concurrent_budget_calls_cannot_overbook_one_reservation(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    calls=[]
    def uncertain(*args):
        calls.append(True)
        raise TimeoutError()
    budget=BudgetTransport(tmp_path/'ledger.json',limit='0.003',transport=uncertain)
    def attempt(_):
        try:
            budget(URL,'synthetic',PAYLOAD,10)
        except ValueError:
            return False
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(attempt,range(4))) == [False]*4
    assert len(calls)==1
    assert budget.charged == RESERVE


@pytest.mark.parametrize('value,expected',[(.9,'yes'),(.1,'no'),(.5,'uncertain'),(True,'unavailable'),(float('nan'),'unavailable'),(None,'unavailable')])
def test_post_run_review_keeps_uncertainty_separate(value,expected):
    from alphonse.agent_v2.evaluation.resourceful_live_review import classify
    assert classify(value)==expected


def test_post_run_review_handles_rejected_call_without_result(tmp_path, monkeypatch):
    import json
    import sys
    from types import SimpleNamespace
    from alphonse.agent_v2.evaluation import resourceful_live_review as review
    source=tmp_path/'source.json';output=tmp_path/'review.json';ledger=tmp_path/'budget.json'
    source.write_text(json.dumps({'revision':'synthetic','run_id':'synthetic','live_model_qualified':False,
        'results':[{'id':'rejected','repeat':1,'split':'heldout','scenario_pass':False,
            'task':{'status':'failed','metadata':{'v3_phase_history':[{'evidence':{'entries':[
                {'evidence_ref':'rejected:1','tool_id':'native.respond','status':'rejected','result':None,'arguments':{}}
            ]}}]}}}]}))
    def send(*args):
        return {'usage':{'input_tokens':10},'answers':{k:{'noul':.5} for k in review.QUESTIONS}}
    monkeypatch.setattr(review,'BudgetTransport',lambda p:BudgetTransport(p,transport=send))
    monkeypatch.setattr(review,'read_settings',lambda p:(None,SimpleNamespace(api_key='synthetic')))
    monkeypatch.setattr(sys,'argv',['review','--input',str(source),'--output',str(output),'--ledger',str(ledger)])
    review.main()
    result=json.loads(output.read_text())
    assert result['results'][0]['metrics']['rejected_tool_proposals']==1
    assert result['results'][0]['metrics']['delivered_question_interrupts']==0
    assert result['qualification_unchanged']
