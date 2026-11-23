import json
from urllib.request import urlopen

from alphonse.agent_v2.evaluation.dojo_host_device import DeviceFixture


def test_device_tracks_real_requests_and_does_not_infer_answer_success():
    with DeviceFixture() as device:
        base = f'http://127.0.0.1:{device.server.server_port}'
        with urlopen(base, timeout=2) as response:
            root = json.load(response)
        assert root['status_url'] == '/status'
        with urlopen(base + root['status_url'], timeout=2) as response:
            status = json.load(response)
        grade = device.grade(device.readback(None), f"Humidity is {status['humidity_percent']}%")
        assert grade['checks']['status_endpoint_read'] is True
        assert grade['checks']['answer_mentions_measurement'] is True
        assert grade['effect_passed'] is None
        assert grade['response_semantics'] == 'requires_manual_review'
        assert device.grade(device.readback(None), 'I cannot check')['checks']['answer_mentions_measurement'] is False
