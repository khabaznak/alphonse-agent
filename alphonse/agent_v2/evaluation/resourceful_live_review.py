"""Post-run diagnostic audit of retained live traces; never changes qualification.

Keeps uncertainty separate from positive inability judgments and recovers raw
counts/provider durations for trials whose provider raised before final scoring.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import subprocess

from .resourceful_live import AUDIT_QUESTIONS, BudgetTransport, read_settings
from alphonse.agent_v2.database import default_database_path

QUESTIONS = dict(AUDIT_QUESTIONS, unnecessary_input_request={
    'type':'noul', 'instructions':'Does response ask for input that goal or authenticated answers already supply, or for a routine implementation choice the user delegated?',
    'criteria':{'true':'It asks the user to repeat supplied information or decide a delegated routine detail.',
                'false':'It asks no question, or asks for genuinely missing essential input or a material scope, privacy, cost or external-effect choice.'}})


def observations(task):
    rows=[e for h in task.get('metadata',{}).get('v3_phase_history',[]) for e in h.get('evidence',{}).get('entries',[])]
    rows+=task.get('hierarchical_state',{}).get('evidence',{}).get('entries',[])
    return list({e['evidence_ref']:e for e in rows}.values())


def classify(probability):
    if type(probability) not in {int,float} or not 0 <= probability <= 1:
        return 'unavailable'
    return 'yes' if probability >= .8 else 'no' if probability <= .2 else 'uncertain'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--ledger',type=Path,required=True)
    args=parser.parse_args()
    raw=args.input.read_bytes();source=json.loads(raw)
    _,settings=read_settings(default_database_path())
    budget=BudgetTransport(args.ledger)
    report={'source_sha256':hashlib.sha256(raw).hexdigest(),'source_revision':source['revision'],
        'review_revision':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'qualification_unchanged':True,'original_live_model_qualified':source['live_model_qualified'],
        'review_kind':'post-hoc diagnostic, not a new qualification or threshold change','results':[]}
    original_calls=list(budget.rows)
    for result in source['results']:
        task=result.get('task',{});metadata=task.get('metadata',{})
        rows=observations(task)
        response=str(metadata.get('prepared_user_response',{}).get('message') or metadata.get('v3_failure_response',{}).get('message') or '')
        case_key=source['run_id']+':'+result['id']+':'+str(result['repeat'])
        calls=[r for r in original_calls if r.get('case_id')==case_key]
        budget.local.case_id='post-review:'+case_key
        payload={'model':'jev-1.14.0','state':{'goal':task.get('goal',''),'response':response,
            'authenticated_answers':metadata.get('v3_resumption_evidence',[]),'observations':rows,
            'http_receipts':result.get('http_receipts',[]),'schedules':result.get('schedules',[])},'questions':QUESTIONS}
        try:
            audit=budget('https://api.typesafe.ai/v1/systemone',settings.api_key,payload,10)
        except Exception as error:
            audit={'answers':{},'error':str(error)}
        classified={key:classify(audit.get('answers',{}).get(key,{}).get('noul')) for key in QUESTIONS}
        rejected=[e for e in rows if e.get('status')=='rejected']
        duplicates=max(0,len(rejected)-len({json.dumps([e.get('tool_id'),e.get('arguments')],sort_keys=True) for e in rejected}))
        report['results'].append({'id':result['id'],'repeat':result['repeat'],'split':result['split'],
            'original_pass':result['scenario_pass'],'error':result.get('error'),'status':task.get('status'),
            'response':response,'audit':audit,'classified':classified,
            'metrics':{'system_two_calls':len(result.get('requests',[])),
                'system_two_seconds':sum(q.get('seconds',0) for q in result.get('requests',[])),
                'typesafe_calls':len(calls),'typesafe_seconds':sum(c.get('seconds',0) for c in calls),
                'typesafe_charge_usd':str(sum((Decimal(c['budget_charge_usd']) for c in calls),Decimal(0))),
                'case_wall_seconds':result.get('metrics',{}).get('latency_seconds'),
                'delivered_question_interrupts':sum(bool(e.get('result',{}).get('question_interrupt')) for e in rows),
                'authenticated_user_interventions':sum(e.get('kind')=='authenticated_question_answer' for e in rows),
                'rejected_tool_proposals':len(rejected),'repeated_equivalent_rejected_proposals':duplicates,
                'repeated_prepared_response_messages':max(0,len([e for e in rows if e.get('tool_id')=='native.respond'])-len({e.get('result',{}).get('message') for e in rows if e.get('tool_id')=='native.respond'}))}})
        report['total_typesafe_budget_charge_usd']=str(budget.charged)
        args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'reviewed':len(report['results']),'total_typesafe_budget_charge_usd':str(budget.charged)}))


if __name__=='__main__':
    main()
