from decimal import Decimal

import pytest

from alphonse.agent_v2.evaluation.resourceful_live import BudgetTransport, RESERVE

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


def test_failed_workflow_retains_metrics_and_uncertain_audit_is_not_an_inability_claim(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from alphonse.agent_v2.evaluation.resourceful_live import LiveTrial
    def send(*args):
        return {'usage':{'input_tokens':100},'answers':{'response_supported':{'noul':.5},'unsupported_inability':{'noul':.5}}}
    budget=BudgetTransport(tmp_path/'ledger.json',transport=send)
    case={'id':'notes-crash','kind':'notes','split':'development','request':'Create notes','expected':'completed'}
    trial=LiveTrial(tmp_path,case,'http://127.0.0.1:1',[],SimpleNamespace(),SimpleNamespace(settings=SimpleNamespace(api_key='synthetic')),budget,'case')
    def fail():
        raise ValueError('synthetic inference error')
    monkeypatch.setattr(trial,'run',fail)
    result=trial.run_safe()
    assert not result['scenario_pass']
    assert result['error']=='synthetic inference error'
    assert result['metrics']['latency_seconds'] > 0
    assert result['metrics']['unsupported_inability_claims']==0
    assert result['metrics']['uncertain_inability_judgments']==1
    assert not result['checks']['no_unsupported_inability']


def test_audit_outage_still_retains_failed_workflow_measurements(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from alphonse.agent_v2.evaluation.resourceful_live import LiveTrial
    def outage(*args):
        raise TimeoutError()
    budget=BudgetTransport(tmp_path/'ledger.json',transport=outage)
    case={'id':'audit-crash','kind':'notes','split':'heldout','request':'Create notes','expected':'completed'}
    trial=LiveTrial(tmp_path,case,'http://127.0.0.1:1',[],SimpleNamespace(),SimpleNamespace(settings=SimpleNamespace(api_key='synthetic')),budget,'case')
    monkeypatch.setattr(trial,'run',lambda:(_ for _ in ()).throw(ValueError('workflow failed')))
    result=trial.run_safe()
    assert not result['scenario_pass']
    assert 'error' in result['response_audit']
    assert result['metrics']['typesafe_cost_usd']=='0.003'


@pytest.mark.parametrize('value,expected',[(.9,'yes'),(.1,'no'),(.5,'uncertain'),(True,'unavailable'),(float('nan'),'unavailable'),(None,'unavailable')])
def test_post_run_review_keeps_uncertainty_separate(value,expected):
    from alphonse.agent_v2.evaluation.resourceful_live_review import classify
    assert classify(value)==expected
