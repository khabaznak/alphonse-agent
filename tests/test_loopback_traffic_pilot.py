"""Local synthetic HTTP fixture checks; not external-agent evidence."""
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import socket
import time

import pytest

from alphonse.agent_v2.evaluation.loopback_traffic_pilot import LoopbackTrafficPortal, PortalConfig, evaluate


def request(url, fields=None):
    data = None if fields is None else urlencode(fields).encode()
    try:
        with urlopen(Request(url, data=data), timeout=2) as response:
            return response.status, response.read().decode()
    except HTTPError as exc:
        return exc.code, exc.read().decode()


def test_browser_form_and_separate_expected_result():
    config = PortalConfig('FAKE-100', {'FAKE-100': ('SYN-1',)})
    with LoopbackTrafficPortal(config) as portal:
        assert portal.url.startswith('http://127.0.0.1:')
        code, page = request(portal.url)
        assert code == 200 and 'name="plate"' in page and 'SYN-1' not in page
        assert request(portal.url + 'audit')[0] == 404
        assert request(portal.url + 'oracle')[0] == 404
        status, body = request(portal.url + 'lookup', {'plate': 'FAKE-100'})
        assert status == 200 and 'SYN-1' in body
        answer = {'status': 'verified', 'subject': 'FAKE-100', 'ticket_ids': ['SYN-1']}
        with pytest.raises(RuntimeError, match='portal_must_be_finalized'):
            evaluate(portal, expected_subject='FAKE-100', expected_ticket_ids=('SYN-1',), response=answer)
    assert evaluate(portal, expected_subject='FAKE-100', expected_ticket_ids=('SYN-1',), response=answer)['passed']
    wrong_key = evaluate(portal, expected_subject='FAKE-100', expected_ticket_ids=('SYN-2',), response=answer)
    assert not wrong_key['passed']
    assert wrong_key['delivery'] == 'unassessed' and not wrong_key['decision_qualified']
    with pytest.raises(RuntimeError, match='portal_cannot_be_reused'):
        with portal:
            pass


def test_cannot_guess_without_observed_lookup_or_access_other_subject():
    answer = {'status': 'verified', 'subject': 'FAKE-100', 'ticket_ids': []}
    with LoopbackTrafficPortal(PortalConfig('FAKE-100', {'FAKE-100': ()})) as empty:
        request(empty.url)
    assert not evaluate(empty, expected_subject='FAKE-100', expected_ticket_ids=(), response=answer)['passed']
    with LoopbackTrafficPortal(PortalConfig('FAKE-100', {'FAKE-100': (), 'OTHER': ('SECRET-TEST',)})) as portal:
        code, body = request(portal.url + 'lookup', {'plate': 'OTHER'})
        assert code == 403 and 'SECRET-TEST' not in body
        assert request(portal.url + 'lookup', {'plate': 'FAKE-100'})[0] == 200
    outcome = evaluate(portal, expected_subject='FAKE-100', expected_ticket_ids=(), response=answer)
    assert outcome['wrong_subject_attempt'] and not outcome['passed']


def test_changed_field_and_verification_block_are_observed():
    config = PortalConfig('FAKE-200', {'FAKE-200': ('SYN-2',)}, field_name='registration', verification_required=True)
    with LoopbackTrafficPortal(config) as portal:
        assert 'name="registration"' in request(portal.url)[1]
        assert request(portal.url + 'lookup', {'plate': 'FAKE-200'})[0] == 400
        assert request(portal.url + 'lookup', {'registration': 'FAKE-200'})[0] == 403
    block = {'status': 'blocked', 'reason': 'verification_required'}
    outcome = evaluate(portal, expected_subject='FAKE-200', expected_ticket_ids=('SYN-2',), response=block)
    assert outcome['passed'] and outcome['observed_verification_block']
    assert not outcome['decision_qualified']


def test_output_escapes_synthetic_values():
    with LoopbackTrafficPortal(PortalConfig('FAKE-300', {'FAKE-300': ('<synthetic>',)})) as portal:
        _, body = request(portal.url + 'lookup', {'plate': 'FAKE-300'})
        assert '&lt;synthetic&gt;' in body and '<synthetic>' not in body


def test_incomplete_request_does_not_turn_observed_lookup_into_pass():
    with LoopbackTrafficPortal(PortalConfig('FAKE-300', {'FAKE-300': ()})) as portal:
        assert request(portal.url + 'lookup', {'plate': 'FAKE-300'})[0] == 200
        port = int(portal.url.split(':')[2].rstrip('/'))
        connection = socket.create_connection(('127.0.0.1', port), timeout=2)
        connection.sendall(b'POST /lookup HTTP/1.1\r\nHost: localhost\r\nContent-Length: 20\r\n\r\na=')
        # Wait until the handler logs its bounded timeout before closing the portal.
        time.sleep(1.2)
        connection.close()
    answer = {'status': 'verified', 'subject': 'FAKE-300', 'ticket_ids': []}
    result = evaluate(portal, expected_subject='FAKE-300', expected_ticket_ids=(), response=answer)
    assert result['incomplete_http_request'] and not result['passed']


def test_missing_subject_records_are_not_interpreted_as_no_tickets():
    with pytest.raises(ValueError, match='invalid_synthetic_configuration'):
        LoopbackTrafficPortal(PortalConfig('FAKE-400', {}))


def test_valid_short_body_with_half_close_is_incomplete():
    with LoopbackTrafficPortal(PortalConfig('FAKE-400', {'FAKE-400': ()})) as portal:
        port = int(portal.url.split(':')[2].rstrip('/'))
        with socket.create_connection(('127.0.0.1', port), timeout=2) as connection:
            body = b'plate=FAKE-400'
            connection.sendall(b'POST /lookup HTTP/1.1\r\nHost: localhost\r\nContent-Length: 30\r\n\r\n' + body)
            connection.shutdown(socket.SHUT_WR)
            assert b'400' in connection.recv(1024)
    answer = {'status': 'verified', 'subject': 'FAKE-400', 'ticket_ids': []}
    result = evaluate(portal, expected_subject='FAKE-400', expected_ticket_ids=(), response=answer)
    assert result['incomplete_http_request'] and not result['passed']
    assert not result['observed_matching_lookup']


def test_unstarted_portal_cannot_be_finalized():
    portal = LoopbackTrafficPortal(PortalConfig('FAKE-400', {'FAKE-400': ()}))
    with pytest.raises(RuntimeError, match='portal_not_started'):
        portal.__exit__(None, None, None)
    with pytest.raises(RuntimeError, match='portal_must_be_finalized'):
        evaluate(portal, expected_subject='FAKE-400', expected_ticket_ids=(), response={})


def test_optional_capability_requires_secret_for_page_and_lookup_and_is_not_audited():
    config = PortalConfig('FAKE-100', {'FAKE-100': ('SYN-1',)}, require_capability=True)
    with LoopbackTrafficPortal(config) as portal:
        base = portal.url.rsplit('/', 2)[0] + '/'
        secret = portal.url.split('/')[-2]
        assert request(base)[0] == 404
        assert request(base + 'lookup', {'plate': 'FAKE-100'})[0] == 404
        assert request(base + 'not-the-capability/lookup', {'plate': 'FAKE-100'})[0] == 404
        status, page = request(portal.url)
        assert status == 200 and 'action="lookup"' in page
        assert secret not in page
        assert request(portal.url + 'lookup', {'plate': 'FAKE-100'})[0] == 200
    response = {'status': 'verified', 'subject': 'FAKE-100', 'ticket_ids': ['SYN-1']}
    result = evaluate(portal, expected_subject='FAKE-100', expected_ticket_ids=('SYN-1',), response=response)
    assert result['passed'] and result['request_attribution'] == 'bearer_capability_only'
    assert not result['decision_qualified']
    assert secret not in repr(portal.audit()) and secret not in repr(result)


def test_each_capability_is_fresh_and_invalid_options_fail():
    config = PortalConfig('FAKE-100', {'FAKE-100': ()}, require_capability=True)
    first, second = LoopbackTrafficPortal(config), LoopbackTrafficPortal(config)
    assert first._capability != second._capability
    with pytest.raises(ValueError, match='invalid_capability_setting'):
        LoopbackTrafficPortal(PortalConfig('FAKE-100', {'FAKE-100': ()}, require_capability='yes'))
