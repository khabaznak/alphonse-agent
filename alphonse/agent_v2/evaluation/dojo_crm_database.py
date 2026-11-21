"""Create a small CRM database from a disposable CSV through the owner worker.

The database is independently read back. A conventional matching schema gets
an update test on an in-memory copy; other schemas require manual review.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys

from alphonse.agent_v2.evaluation.owner_dojo_shopping import run

CASE_ID = 'prospect-crm-database'
GOAL = ('I have prospects in /workspace/household/prospects.csv. Please create a small SQLite CRM '
        'at /workspace/household/prospects.db with their IDs, names, companies and statuses, '
        'so I can manage these leads later. Keep the original CSV unchanged and tell me when it is ready.')
CSV = ('id,name,company,status\n'
       'P101,Asha Patel,Acorn Systems,new\n'
       'P102,Beatriz Cruz,Cedar Labs,qualified\n'
       'P103,Devin Shaw,Juniper Works,contacted\n'
       'P104,Eli Chen,Larch Studio,new\n')
LEADS = (
    ('P101', 'Asha Patel', 'Acorn Systems', 'new'),
    ('P102', 'Beatriz Cruz', 'Cedar Labs', 'qualified'),
    ('P103', 'Devin Shaw', 'Juniper Works', 'contacted'),
    ('P104', 'Eli Chen', 'Larch Studio', 'new'),
)


def seed(shell):
    program = ('import pathlib; p=pathlib.Path("/workspace/household"); p.mkdir(); '
               '(p/"prospects.csv").write_text(%r)' % CSV)
    result = shell.execute('/usr/local/bin/python -I -c ' + shlex.quote(program))
    if result['status'] != 'completed' or result['exit_code'] != 0:
        raise RuntimeError('crm_fixture_setup_failed')


def readback(shell):
    # Bounded, read-only SQLite inspection. Never invoke SQL supplied by the
    # model; quote discovered table names and do not follow the output symlink.
    program = '''import hashlib,json,os,pathlib,sqlite3,stat
p=pathlib.Path('/workspace/household'); source=p/'prospects.csv'; target=p/'prospects.db'
result={'status':'observed','source_sha256':None,'database_status':'unknown','tables':{},'scan_complete':False}
try:
 if source.is_symlink() or not source.is_file() or source.stat().st_size > 16384: raise ValueError()
 result['source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
except (OSError,ValueError): pass
try:
 st=target.lstat()
 if not stat.S_ISREG(st.st_mode) or st.st_size > 4194304: raise ValueError()
 conn=sqlite3.connect('file:'+str(target)+'?mode=ro',uri=True)
 conn.execute('PRAGMA query_only=ON')
 copy=sqlite3.connect(':memory:'); conn.backup(copy)
 names=conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name LIMIT 101").fetchall()
 if len(names)>100: raise ValueError()
 complete=True
 for (name,) in names:
  columns=conn.execute('PRAGMA table_info("'+name.replace('"','""')+'")').fetchall()
  if len(columns)>50: complete=False; break
  rows=conn.execute('SELECT * FROM "'+name.replace('"','""')+'" LIMIT 201').fetchall()
  if len(rows)>200: complete=False; rows=rows[:200]
  colnames=[r[1] for r in columns]
  probe=None
  if all(sum(isinstance(col,str) and col.casefold()==need for col in colnames)==1 for need in ('id','name','company','status')):
   try:
    quoted='"'+name.replace('"','""')+'"'
    copy.execute('BEGIN')
    changed=copy.execute('UPDATE '+quoted+' SET "status"=? WHERE "id"=?',('qualified','P101')).rowcount
    after=copy.execute('SELECT "status" FROM '+quoted+' WHERE "id"=?',('P101',)).fetchall()
    probe=changed==1 and after==[('qualified',)]
   except sqlite3.Error: probe=False
   finally: copy.rollback()
  result['tables'][name]={'columns':colnames, 'rows':[[v if isinstance(v,(str,int,float)) or v is None else '<binary>' for v in row] for row in rows], 'update_probe':probe}
 copy.close(); conn.close(); result['database_status']='observed'; result['scan_complete']=complete
except FileNotFoundError: result['database_status']='missing'
except (OSError,ValueError,sqlite3.Error): result['database_status']='invalid_or_unreadable'
print(json.dumps(result,ensure_ascii=False))
'''
    result = shell.execute('/usr/local/bin/python -I -c ' + shlex.quote(program))
    if result['status'] != 'completed' or result['exit_code'] != 0 or result['truncated']:
        return {'status': 'unavailable'}
    try:
        observation = json.loads(result['stdout'])
        return observation if isinstance(observation, dict) else {'status': 'unavailable'}
    except (ValueError, TypeError):
        return {'status': 'unavailable'}


def grade(observation):
    if not isinstance(observation, dict) or observation.get('status') != 'observed':
        return {'effect_passed': None, 'reason': 'readback_unavailable', 'response_semantics': 'unassessed'}
    source_hash = observation.get('source_sha256')
    unchanged = None if source_hash is None else source_hash == hashlib.sha256(CSV.encode()).hexdigest()
    db_status = observation.get('database_status')
    complete = observation.get('scan_complete') is True
    tables = observation.get('tables')
    rows = [row for table in tables.values() if isinstance(table, dict)
            for row in table.get('rows', []) if isinstance(row, list)] if isinstance(tables, dict) else []
    # Same-row presence is useful evidence, but does not establish a usable
    # schema, correctness of the complete import, or future manageability.
    matched = sum(any(all(field in row for field in lead) for row in rows) for lead in LEADS)
    conventional = []
    if isinstance(tables, dict):
        for name, table in tables.items():
            if not isinstance(table, dict):
                continue
            columns = table.get('columns')
            table_rows = table.get('rows')
            if not isinstance(columns, list) or not isinstance(table_rows, list):
                continue
            if any(sum(isinstance(col, str) and col.casefold() == key for col in columns) != 1
                   for key in ('id', 'name', 'company', 'status')):
                continue
            indexes = [next(i for i, col in enumerate(columns) if isinstance(col, str) and col.casefold() == key)
                       for key in ('id', 'name', 'company', 'status')]
            if any(not isinstance(row, list) or len(row) != len(columns) for row in table_rows):
                continue
            records = [tuple(row[i] for i in indexes) for row in table_rows]
            exact = len(records) == len(LEADS) and all(records.count(lead) == 1 for lead in LEADS)
            conventional.append({'table': name, 'exact_source_rows': exact,
                                 'update_probe': table.get('update_probe')})
    verified = [item['table'] for item in conventional
                if item['exact_source_rows'] and item['update_probe'] is True]
    if unchanged is False or db_status in {'missing', 'invalid_or_unreadable'}:
        effect = False
    elif unchanged is True and db_status == 'observed' and complete and verified:
        effect = True
    else:
        effect = None
    return {'effect_passed': effect, 'database_status': db_status,
            'source_unchanged': unchanged, 'expected_leads_with_same_row_evidence': matched,
            'scan_complete': complete, 'conventional_table_checks': conventional,
            'verified_import_and_update_tables': verified,
            'schema_interpretation': 'manual_review_if_not_verified',
            'response_semantics': 'unassessed'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', type=Path)
    parser.add_argument('--image-id', required=True)
    parser.add_argument('--docker-socket', required=True)
    parser.add_argument('--model', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(str(path.relative_to(root)).encode() + b'\0' + path.read_bytes() + b'\0')
    with args.journal.open('x', encoding='utf-8') as stream:
        def emit(event):
            stream.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
        emit({'event': 'started', 'case': CASE_ID, 'goal': GOAL, 'model': args.model,
              'image_id': args.image_id, 'source_tree_sha256': digest.hexdigest(),
              'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'synthetic owner worker; disposable CSV and container Bash'})
        try:
            result = run(image_id=args.image_id, docker_socket=args.docker_socket, emit=emit,
                         model=args.model, timeout_seconds=120, scenario=sys.modules[__name__], full_registry=True)
        except Exception as exc:
            emit({'event': 'finished', 'outcome': 'infrastructure_error', 'task_passed': None,
                  'error_type': type(exc).__name__})
            raise
        result['limitations'].extend([
            'Injected container Bash is not normal owner host Bash.',
            'Positive grading means a conventional table matched all source rows and an update worked on a disposable in-memory copy; it does not prove full CRM functionality.',
            'Normalized schemas, extra records, custom constraints and absent probes need manual review.',
            'Final CSV hash does not prove no transient modification occurred; response semantics are unassessed.',
        ])
        emit({'event': 'finished', **result})
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
