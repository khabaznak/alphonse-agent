"""Independent receiving inbox for the disposable mixed-follow-up Dojo."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys


_RECEIVE = """
import json
from pathlib import Path
import sys

request = json.load(sys.stdin)
envelope = {'request': request, 'status': 'accepted'}
with Path(sys.argv[1]).open('x', encoding='utf-8') as receipt:
    json.dump(envelope, receipt)
print('accepted')
"""


class DisposableReceiver:
    """A separate process durably accepts the brief without applying a change."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir()

    def receipts(self):
        return [json.loads(receipt.read_text(encoding='utf-8'))
                for receipt in sorted(self.root.glob('*.json'))]

    def _receipt_path(self, request):
        key = hashlib.sha256(request.operation_id.encode()).hexdigest()
        return self.root / f'{key}.json'

    def submit(self, request, claim_active):
        if claim_active() is not True:
            return None
        completed = subprocess.run(
            [sys.executable, '-I', '-B', '-c', _RECEIVE, str(self._receipt_path(request))],
            input=json.dumps(asdict(request)), encoding='utf-8', capture_output=True, timeout=8,
            cwd=self.root, env={},
        )
        if completed.returncode != 0 or completed.stdout.strip() != 'accepted':
            return None
        return self.reconcile(request, claim_active)

    def reconcile(self, request, claim_active):
        from alphonse.agent_v2.core.intelligence.simple_loop.owner_disposable_handoff import (
            DisposableHandoffReceipt,
        )

        if claim_active() is not True:
            return None
        try:
            stored = json.loads(self._receipt_path(request).read_text(encoding='utf-8'))
        except FileNotFoundError:
            return None
        if stored != {'request': asdict(request), 'status': 'accepted'}:
            return None
        return DisposableHandoffReceipt(request, 'accepted')


def attach_disposable_receiver(runtime, *, actor, target, revision, prompt, receiver):
    from alphonse.agent_v2.core.intelligence.simple_loop.owner_disposable_handoff import (
        DisposableHandoffBinding, DisposableHandoffService,
    )

    binding = DisposableHandoffBinding(
        actor=actor, target=target, revision=revision, operation='propose_patch',
        owner_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
    )
    service = DisposableHandoffService(
        runtime.queue.db_path, bindings=(binding,), submit=receiver.submit,
        reconcile=receiver.reconcile, verify_result=getattr(receiver, 'verify_result', lambda *_: False),
    )
    runtime.processor.owner_read_candidate.enable_dojo_handoff(service)
    return service
