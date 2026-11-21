from types import SimpleNamespace

from alphonse.agent_v2.core.intelligence.simple_loop.owner_read_processor import _owner_shell_should_stop


def test_running_shell_stops_on_cancel_revocation_or_lost_exact_claim():
    current = {'cancelled': False, 'authorized': True, 'claim': True}
    context = SimpleNamespace(is_cancelled=lambda: current['cancelled'])
    def check_claim(reservation_id, kind, *, tool_id, arguments):
        assert (reservation_id, kind, tool_id, arguments) == ('action-1', 'tool', 'native.bash', {'command': 'true'})
        return current['claim']
    atomic = SimpleNamespace(check_claim=check_claim)
    state = SimpleNamespace(inflight='action-1')
    stop = lambda: _owner_shell_should_stop(context, atomic, state, 'owner', 'native.bash',
                                            {'command': 'true'}, lambda *_: current['authorized'])
    assert stop() is False
    for key in current:
        current[key] = key == 'cancelled'
        if key != 'cancelled':
            current[key] = False
        assert stop() is True
        current.update(cancelled=False, authorized=True, claim=True)
    state.inflight = ''
    assert stop() is True


def test_shell_monitor_fails_closed_on_exception():
    def denied(*_args, **_kwargs): raise RuntimeError('private detail')
    context = SimpleNamespace(is_cancelled=lambda: False)
    assert _owner_shell_should_stop(context, SimpleNamespace(check_claim=denied),
        SimpleNamespace(inflight='action-1'), 'owner', 'native.bash', {}, lambda *_: True)
