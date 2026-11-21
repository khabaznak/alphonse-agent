import ctypes
import os
import sys

import pytest

from alphonse.agent_v2.authority_service.macos import launchd_activation as activation


class Function:
    def __init__(self, callback):
        self.callback = callback
    def __call__(self, *args):
        return self.callback(*args)


@pytest.mark.skipif(sys.platform != 'darwin', reason='macOS launchd')
def test_real_non_launchd_process_fails_without_starting_a_service():
    with pytest.raises(activation.LaunchdActivationError):
        activation._activate('Desktop')


@pytest.mark.skipif(sys.platform != 'darwin', reason='macOS launchd ABI')
def test_single_descriptor_is_returned_and_array_is_freed(monkeypatch):
    read_fd, write_fd = os.pipe()
    array = (ctypes.c_int * 1)(read_fd)
    freed = []
    def activate(name, output, count):
        assert name == b'Desktop'
        ctypes.cast(output, ctypes.POINTER(ctypes.POINTER(ctypes.c_int)))[0] = ctypes.cast(array, ctypes.POINTER(ctypes.c_int))
        ctypes.cast(count, ctypes.POINTER(ctypes.c_size_t))[0] = 1
        return 0
    class Library:
        launch_activate_socket = Function(activate)
        free = Function(lambda pointer: freed.append(bool(pointer)))
    monkeypatch.setattr(activation, '_library', lambda: Library())
    try:
        assert activation._activate('Desktop') == read_fd
        assert not os.get_inheritable(read_fd)
        assert freed == [True]
    finally:
        os.close(read_fd)
        os.close(write_fd)


@pytest.mark.skipif(sys.platform != 'darwin', reason='macOS launchd ABI')
def test_multiple_descriptors_are_closed_and_rejected(monkeypatch):
    first, first_write = os.pipe()
    second, second_write = os.pipe()
    array = (ctypes.c_int * 2)(first, second)
    freed = []
    def activate(_name, output, count):
        ctypes.cast(output, ctypes.POINTER(ctypes.POINTER(ctypes.c_int)))[0] = ctypes.cast(array, ctypes.POINTER(ctypes.c_int))
        ctypes.cast(count, ctypes.POINTER(ctypes.c_size_t))[0] = 2
        return 0
    class Library:
        launch_activate_socket = Function(activate)
        free = Function(lambda pointer: freed.append(bool(pointer)))
    monkeypatch.setattr(activation, '_library', lambda: Library())
    try:
        with pytest.raises(activation.LaunchdActivationError):
            activation._activate('Worker')
        for fd in (first, second):
            with pytest.raises(OSError):
                os.fstat(fd)
        assert freed == [True]
    finally:
        os.close(first_write)
        os.close(second_write)


def test_second_activation_failure_closes_first_descriptor(monkeypatch):
    read_fd, write_fd = os.pipe()
    def activate(name):
        if name == 'Desktop':
            return read_fd
        raise activation.LaunchdActivationError('unavailable')
    monkeypatch.setattr(activation, '_activate', activate)
    try:
        with pytest.raises(activation.LaunchdActivationError):
            with activation.activated_listeners():
                pytest.fail('second activation must fail')
        with pytest.raises(OSError):
            os.fstat(read_fd)
    finally:
        os.close(write_fd)
