"""Wire the synthetic candidate to Responses request/usage plumbing offline.

The sender MUST be a declared offline test double; the declaration is not an
OS or network sandbox. The direct network sender is rejected, but a callable
could falsely declare itself offline and delegate to a network sender. This
module is not a live-pilot approval, isolation boundary or billing guarantee.
It exercises the request, ledger and candidate contracts using synthetic data.
"""
from __future__ import annotations

from .model_candidate_bridge import CandidateDecisionBridge
from .responses_pilot_transport import PilotLedger, ResponsesPilotTransport, requests_sender


class OfflineResponsesTransport:
    """Adapt Responses' pilot transport to an explicitly offline sender."""

    offline = True

    def __init__(self, *, model: str, ledger: PilotLedger, counter, sender):
        if sender is requests_sender or getattr(sender, 'offline', None) is not True:
            raise ValueError('offline_sender_required')
        # The dummy key is never a real credential; an injected double still
        # must not send network traffic. Assertions cannot enforce isolation.
        self._transport = ResponsesPilotTransport(model=model, api_key='offline-test-only',
            ledger=ledger, count_complete_input=counter, sender=sender, enabled=True)

    def generate_json(self, request, *, max_input_tokens, max_output_tokens):
        return self._transport.generate_json(request, max_input_tokens=max_input_tokens,
                                             max_output_tokens=max_output_tokens)


def candidate_chooser(*, model: str, ledger: PilotLedger, counter, sender,
                      emit, max_input_tokens: int,
                      allowed_source_names: tuple[str, ...] = ('index', 'conversation', 'prescription', 'treatment-status')):
    """Create an offline fixture chooser; host authorization remains separate."""
    transport = OfflineResponsesTransport(model=model, ledger=ledger,
                                          counter=counter, sender=sender)
    return CandidateDecisionBridge(transport, max_input_tokens, emit,
                                   fixture_argument='source', allowed_source_names=allowed_source_names)
