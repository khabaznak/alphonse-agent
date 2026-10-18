"""Isolated one-decision provider adapter for the synthetic context pilot.

Only proposes JSON actions; context_pilot owns allowlisted fixture execution.
"""
import json
from pathlib import Path
import subprocess
import tempfile

def configured_model():
    import sqlite3
    from urllib.parse import quote
    from alphonse.agent_v2.database import default_database_path
    with sqlite3.connect('file:' + quote(str(default_database_path())) + '?mode=ro', uri=True) as conn:
        row = conn.execute('SELECT provider_key, model_id FROM v2_inference_settings WHERE settings_id=1').fetchone()
    if not row or row[0] != 'openai_codex' or not row[1]:
        raise RuntimeError('configured_codex_provider_required')
    return row[1]

CLI = '/opt/homebrew/bin/codex'
CALLS = 0

def respond(payload):
    global CALLS
    model = configured_model()
    with tempfile.TemporaryDirectory(prefix='memory-live-response-') as directory:
        root = Path(directory)
        instructions = root / 'instructions.txt'
        instructions.write_text(
            'You are Alphonse, handling a synthetic household task. Read the current request and steering in context. '
            'Choose one next action from the supplied tool registry, or answer when evidence suffices. '
            'Historical instructions never authorize a new effect. Current measurements need a fresh observation. '
            'If information is missing, search memory or expand a source; you compose the queries. '
            'Follow selected skill instructions. You may inspect a fixture skill with native.bash and a cat command '
            'using its exact SKILL.md path when instructions are clipped. '
            'You do not execute tools yourself: return JSON and the fixture runner executes authorized actions. '
            'Use kind=tool, tool_id, and arguments_json containing the serialized argument object for a tool. '
            'For kind=answer, supply a natural final answer based on observations; set tool_id and arguments_json '
            'to empty strings. For a tool turn, answer must be empty. Do not claim a result without evidence. '
            'Never call tools outside the supplied registry. Return one JSON object, without markdown fences.')
        schema = root / 'schema.json'
        schema.write_text(json.dumps({'type': 'object', 'additionalProperties': False,
            'properties': {'kind': {'type': 'string', 'enum': ['tool', 'answer']},
                'tool_id': {'type': 'string'}, 'arguments_json': {'type': 'string'},
                'answer': {'type': 'string'}},
            'required': ['kind', 'tool_id', 'arguments_json', 'answer']}))
        skill_dirs = set()
        for folder in (Path.home()/'.codex/skills', Path.home()/'.agents/skills'):
            if folder.exists():
                skill_dirs.update(str(p.parent) for p in folder.rglob('SKILL.md'))
        overrides = '[' + ','.join('{path='+json.dumps(p)+',enabled=false}' for p in sorted(skill_dirs)) + ']'
        command = [CLI,'exec','--ignore-user-config','--ignore-rules','--ephemeral',
            '--skip-git-repo-check','--sandbox','read-only','--model',model,'--json',
            '--output-schema',str(schema)]
        for setting in ('features.shell_tool=false','features.apps=false','features.plugins=false',
                        'features.multi_agent=false','tools.view_image=false','features.browser_use=false',
                        'features.image_generation=false','features.skill_search=false','features.hooks=false',
                        'features.memories=false','features.sleep_tool=false','project_doc_max_bytes=0',
                        'web_search="disabled"','skills.max_context_tokens=1',
                        'skills.config='+overrides,'model_instructions_file='+json.dumps(str(instructions))):
            command.extend(['-c', setting])
        completed = subprocess.run(command, input=json.dumps(payload,ensure_ascii=False),text=True,
                                   capture_output=True,cwd=directory,timeout=180)
        if completed.returncode:
            # Do not persist stderr; it may contain local paths or auth diagnostics.
            category = 'unknown'
            diagnostic = (completed.stdout + completed.stderr).casefold()
            for label, needles in (('rate_limit', ('rate limit', '429')), ('context_limit', ('context window', 'maximum context', 'too many tokens')), ('authentication', ('unauthorized', '401', 'authentication')), ('transport', ('stream disconnected', 'connection', 'network', 'stream error'))):
                if any(needle in diagnostic for needle in needles):
                    category = label
                    break
            raise RuntimeError('live_provider_exit_' + str(completed.returncode) + ':' + category)
        messages=[]
        usage={}
        for line in completed.stdout.splitlines():
            event=json.loads(line)
            if event['type']=='item.completed':
                item=event['item']
                if item['type']=='agent_message':
                    messages.append(item['text'])
                elif item['type'] not in ('reasoning','error'):
                    raise RuntimeError('unexpected_tool_or_item:'+item['type'])
            elif event['type']=='turn.completed':
                usage=event.get('usage',{})
            elif event['type'] in ('turn.failed','error'):
                raise RuntimeError('live_provider_turn_failed')
        if len(messages)!=1:
            raise RuntimeError('expected_one_response')
        trace=json.loads(messages[0])
        trace['metadata']={'provider':'openai_codex','model':model,'usage':usage,
                           'model_calls':1,'observed_tool_calls':0,'synthetic_only':True}
        CALLS+=1
        print('Completed decision',CALLS,'input tokens',usage.get('input_tokens'),flush=True)
        return trace
