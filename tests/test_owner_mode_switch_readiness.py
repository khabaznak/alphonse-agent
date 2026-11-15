"""Owner mode must not accept work it cannot acquire after a mode switch."""
from types import SimpleNamespace

from alphonse.agent_v2.daemon import V2Daemon


def test_owner_readiness_fails_closed_when_queue_is_unavailable():
    def unavailable():
        raise OSError('fixture database unavailable')

    daemon = SimpleNamespace(experimental_owner_only=True,
        _processor_thread=SimpleNamespace(is_alive=lambda: True),
        _stop=SimpleNamespace(is_set=lambda: False),
        runtime=SimpleNamespace(queue=SimpleNamespace(claim_mode_blocked=unavailable)))
    assert V2Daemon.owner_processing_blocking_reason(daemon) == 'queue_unavailable'
    daemon.owner_processing_blocking_reason = lambda: V2Daemon.owner_processing_blocking_reason(daemon)
    assert V2Daemon.owner_processing_ready(daemon) is False
    assert V2Daemon.owner_paused_legacy_requests(daemon) is None
