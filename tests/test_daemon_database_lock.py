"""Disposable tests for cooperating-daemon exclusion by database path."""
import sqlite3
import os
import threading
from types import SimpleNamespace

import pytest

from alphonse.agent_v2 import daemon as module


def test_database_lock_fences_different_socket_paths_and_symlink_aliases(tmp_path, monkeypatch):
    db = tmp_path / 'household.sqlite'
    sqlite3.connect(db).close()
    alias = tmp_path / 'alias.sqlite'
    alias.symlink_to(db)
    monkeypatch.setenv('ALPHONSE_V2_SOCKET_PATH', str(tmp_path / 'one.sock'))
    socket_one = module._claim_single_instance_lock('one')
    database_one = module._claim_database_instance_lock(str(db), 'one')
    try:
        monkeypatch.setenv('ALPHONSE_V2_SOCKET_PATH', str(tmp_path / 'two.sock'))
        socket_two = module._claim_single_instance_lock('two')
        try:
            with pytest.raises(RuntimeError, match='database_lock_held'):
                module._claim_database_instance_lock(str(alias), 'two')
        finally:
            module._unlock_instance_file(socket_two)
    finally:
        module._unlock_instance_file(database_one)
        module._unlock_instance_file(socket_one)
    second = module._claim_database_instance_lock(str(alias), 'two')
    module._unlock_instance_file(second)


def test_database_lock_conflict_blocks_migration_and_releases_socket_lock(tmp_path, monkeypatch):
    db = tmp_path / 'household.sqlite'
    socket = tmp_path / 'daemon.sock'
    monkeypatch.setattr(module, 'default_database_path', lambda: db)
    monkeypatch.setattr(module, 'default_socket_path', lambda: socket)
    called = []
    monkeypatch.setattr(module, 'migrate_legacy_databases', lambda: called.append(True))
    existing = module._claim_database_instance_lock(str(db), 'existing')
    try:
        with pytest.raises(RuntimeError, match='database_lock_held'):
            module.main()
        assert called == []
        # Startup unwound the socket lock acquired before the DB conflict.
        socket_lock = module._claim_single_instance_lock('next')
        module._unlock_instance_file(socket_lock)
    finally:
        module._unlock_instance_file(existing)


@pytest.mark.parametrize('kind', ['database', 'socket'])
@pytest.mark.parametrize('alias', ['symlink', 'hardlink'])
def test_lock_refuses_linked_targets_without_changing_them(tmp_path, monkeypatch, kind, alias):
    target = tmp_path / 'protected'
    target.write_bytes(b'household data')
    db = tmp_path / 'db.sqlite'
    socket = tmp_path / 'daemon.sock'
    lock = tmp_path / ('db.sqlite.alphonse-daemon.lock' if kind == 'database' else 'daemon.lock')
    if alias == 'symlink':
        lock.symlink_to(target)
    else:
        os.link(target, lock)
    monkeypatch.setattr(module, 'default_socket_path', lambda: socket)
    with pytest.raises((OSError, ValueError)):
        if kind == 'database':
            module._claim_database_instance_lock(str(db), 'fixture')
        else:
            module._claim_single_instance_lock('fixture')
    assert target.read_bytes() == b'household data'


def test_lock_never_overwrites_existing_regular_file(tmp_path):
    db = tmp_path / 'db.sqlite'
    lock = tmp_path / 'db.sqlite.alphonse-daemon.lock'
    lock.write_bytes(b'preserve existing content')
    acquired = module._claim_database_instance_lock(str(db), 'fixture')
    module._unlock_instance_file(acquired)
    assert lock.read_bytes() == b'preserve existing content'


@pytest.mark.parametrize('kind', ['database', 'socket'])
def test_lock_closes_descriptor_if_flock_fails(tmp_path, monkeypatch, kind):
    handle = SimpleNamespace(fileno=lambda: 123, closed=False)
    def close():
        handle.closed = True
    handle.close = close
    monkeypatch.setattr(module, '_open_instance_lock', lambda _: handle)
    monkeypatch.setattr(module.fcntl, 'flock', lambda *_: (_ for _ in ()).throw(OSError('synthetic flock failure')))
    with pytest.raises(OSError, match='synthetic flock failure'):
        if kind == 'database':
            module._claim_database_instance_lock(str(tmp_path / 'db.sqlite'), 'fixture')
        else:
            module._claim_single_instance_lock('fixture')
    assert handle.closed


def test_release_attempts_both_locks_even_when_one_unlock_fails(monkeypatch):
    instance = module.V2Daemon.__new__(module.V2Daemon)
    instance._lifecycle_lock = threading.RLock()
    instance._database_lock_file = 'database'
    instance._lock_file = 'socket'
    attempted = []
    def unlock(lock):
        attempted.append(lock)
        if lock == 'database':
            raise OSError('synthetic failure')
    monkeypatch.setattr(module, '_unlock_instance_file', unlock)
    with pytest.raises(OSError, match='synthetic failure'):
        instance._release_single_instance_lock()
    assert attempted == ['database', 'socket']


def test_database_lock_is_held_until_accepted_requests_drain(tmp_path, monkeypatch):
    db = tmp_path / 'db.sqlite'
    socket = tmp_path / 'daemon.sock'
    monkeypatch.setattr(module, 'default_socket_path', lambda: socket)
    instance = module.V2Daemon.__new__(module.V2Daemon)
    instance.experimental_owner_only = True
    instance._lifecycle_lock = threading.RLock()
    instance._stopped = False
    instance._stop = threading.Event()
    instance._processor_thread = None
    instance._lock_file = module._claim_single_instance_lock('first')
    instance._database_lock_file = module._claim_database_instance_lock(str(db), 'first')
    drained, released = threading.Event(), threading.Event()
    instance.ipc = SimpleNamespace(stop=lambda: None, wait_quiescent=lambda timeout=None: drained.wait(timeout))
    original = instance._release_single_instance_lock
    def release():
        original()
        released.set()
    monkeypatch.setattr(instance, '_release_single_instance_lock', release)
    try:
        instance.stop()
        with pytest.raises(RuntimeError, match='database_lock_held'):
            module._claim_database_instance_lock(str(db), 'second')
        drained.set()
        assert released.wait(3)
        second = module._claim_database_instance_lock(str(db), 'second')
        module._unlock_instance_file(second)
    finally:
        drained.set()
        released.wait(3)
