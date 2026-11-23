import json
from types import SimpleNamespace
from alphonse.agent_v2.evaluation.owner_dojo_shopping import run
scenario = SimpleNamespace(GOAL='Hi Alphonse', CASE_ID='greeting-diagnostic', seed=lambda shell: None, readback=lambda shell: {}, grade=lambda readback: {'effect_passed': None}, record_decision_text=True)
def emit(item): print(json.dumps(item), flush=True)
result = run(image_id='sha256:9c900dea9e8fb7e16277c179b555cc72d29a352dbc33cff48ad5a0412fd5bfc7', docker_socket='/Users/alex/.docker/run/docker.sock', emit=emit, model='gpt-6-luna', timeout_seconds=90, scenario=scenario, full_registry=True, socket_authority=True)
print(json.dumps({'event':'result', **result}), flush=True)
