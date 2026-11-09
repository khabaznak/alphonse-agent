import json

from alphonse.agent_v2.evaluation.browser_submission_probe import Fixture, run_case


def test_submission_oracle_and_v3_completion_are_distinct():
    result = run_case()
    assert result["oracle_result"] == "verified"
    assert result["submission_verified"] and result["verified_fields"] == 2
    assert not result["unresolved_effect"]
    assert result["task_outcome"] == result["v3_completion_guard"] == "blocked"
    assert result["spans"]["browser.submit"]["count"] == 1
    assert result["spans"]["browser.verify_submission"]["count"] == 1
    for value in ("SYNTHETIC_NAME", "SYNTHETIC_NUMBER", "1_1", "fixture-owner"):
        assert value not in json.dumps(result)


def test_changed_form_and_bad_result_do_not_create_verified_task():
    changed = run_case(changed=True)
    assert changed["task_outcome"] == "blocked" and changed["oracle_result"] == "unverified"
    assert changed["unresolved_effect"] and not changed["submission_verified"]
    bad = run_case(bad_result=True)
    assert bad["task_outcome"] == "blocked" and bad["oracle_result"] == "unverified"
    assert bad["unresolved_effect"] and not bad["submission_verified"]
    assert bad["spans"]["browser.verify_submission"]["failed"] == 1


def test_cancelled_attempt_does_not_promote_external_result_to_completion():
    result = run_case(cancelled=True)
    assert result["task_outcome"] == "cancelled" and result["unresolved_effect"]
    assert not result["submission_verified"]


def test_oracle_requires_both_exact_fields_and_single_submission():
    form = Fixture()
    form.submissions = [{"1_1": "SYNTHETIC_NAME"}]
    assert not form.oracle()
    form.submissions = [{"1_1": "SYNTHETIC_NAME", "1_2": "SYNTHETIC_NUMBER"}]
    assert form.oracle()
    form.submissions.append(dict(form.submissions[0]))
    assert not form.oracle()


def test_loopback_site_runs_v3_and_independent_form_with_same_oracle():
    import threading
    from urllib.parse import urlencode
    from urllib.request import Request, urlopen
    import pytest
    from alphonse.agent_v2.evaluation.browser_fixture_site import Site, SiteTransport

    try:
        site = Site(("127.0.0.1", 0))
    except PermissionError:
        pytest.skip("loopback sockets unavailable")
    thread = threading.Thread(target=site.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{site.server_port}"
    try:
        result = run_case(fixture=SiteTransport(base, "v3"))
        assert result["oracle_result"] == "verified" and result["task_outcome"] == "blocked"
        request = Request(base + "/form?run=direct", data=urlencode({"name": "SYNTHETIC_NAME", "number": "SYNTHETIC_NUMBER"}).encode())
        with urlopen(request) as response:
            assert b"Result: accepted" in response.read()
        with urlopen(base + "/oracle?run=direct") as response:
            assert json.load(response) == {"verified": True, "submissions": 1}
        # A duplicate POST cannot rewrite the result or submit a second time.
        replay = Request(base + "/form?run=direct", data=urlencode({"name": "wrong", "number": "wrong"}).encode())
        with urlopen(replay) as response:
            assert b"Result: accepted" in response.read()
        with urlopen(base + "/oracle?run=direct") as response:
            assert json.load(response) == {"verified": True, "submissions": 1}
    finally:
        site.shutdown()
        site.server_close()
        thread.join()
