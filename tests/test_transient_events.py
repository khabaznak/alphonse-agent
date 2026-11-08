from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from alphonse.agent_v2.core.core import CoreUiEvent
from alphonse.agent_v2.transient_events import TransientEvents


def test_concurrent_producers_and_pruning_preserve_fresh_events():
    events = TransientEvents()
    now = datetime.now(timezone.utc)
    old = (now - timedelta(days=31)).isoformat()
    events.extend(CoreUiEvent('state_snapshot', {'id': -1}, occurred_at=old) for _ in range(50))

    def produce():
        for item in range(1000):
            events.append(CoreUiEvent('state_snapshot', {'id': item}))

    def prune():
        for _ in range(100):
            events.prune(now=datetime.now(timezone.utc))

    with ThreadPoolExecutor(max_workers=2) as pool:
        tasks = [pool.submit(produce), pool.submit(prune)]
        for task in tasks:
            task.result()
    assert sorted(event.payload['id'] for event in events.drain()) == list(range(1000))


def test_future_event_age_is_bounded_by_arrival():
    events = TransientEvents()
    now = datetime.now(timezone.utc)
    events.append(CoreUiEvent('state_snapshot', {}, occurred_at=(now + timedelta(days=365)).isoformat()))
    events.prune(now=now + timedelta(days=31))
    assert not events
