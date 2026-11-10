import pytest

from alphonse.agent_v2.core.intelligence.v3.evidence_projection import (
    EvidenceAccessError, EvidenceSource, project_evidence,
)
from alphonse.agent_v2.core.intelligence.v3.projections import ContextBudgetExceeded


def make_projection(sources, data, *, budget=1500, authorize=None):
    return project_evidence(sources, actor="user-a", budget=budget,
        authorize=authorize or (lambda actor, ref: actor == "user-a" and ref in data),
        load=lambda actor, ref: data[ref])


def test_bounded_growing_history_keeps_goal_current_effect_and_reports_omissions():
    goal = EvidenceSource("task:goal", "goal")
    current = EvidenceSource("action:current", "current_result")
    effect = EvidenceSource("effect:unconfirmed", "effect")
    for count in (2, 20, 100):
        observations = [EvidenceSource(f"search:{i}", "observation") for i in range(count)]
        data = {"task:goal": "find count", "action:current": "result partial", "effect:unconfirmed": "uncertain write"}
        data.update({f"search:{i}": "repeat result " + "x" * 150 for i in range(count)})
        p = make_projection([*observations, current, effect, goal], data, budget=3000)
        assert len(p.text.encode("utf-8")) <= 3000
        assert all(data[key] in p.text for key in ("task:goal", "action:current", "effect:unconfirmed"))
        assert p.manifest["coverage"]["omitted_count"] == len(p.manifest["omitted_source_ids"])
        assert p.manifest["coverage"]["absence_is_not_evidence"] is True
        assert "uncertain write" not in str(p.manifest)
        if count == 100:
            assert p.manifest["coverage"]["omitted_count"] > 0


def test_mixed_visibility_never_loads_or_exposes_denied_identifier():
    loads = []
    refs = [EvidenceSource("goal:1", "goal"), EvidenceSource("secret:other", "observation")]
    result = project_evidence(refs, actor="alice", budget=1000,
        authorize=lambda actor, ref: ref == "goal:1",
        load=lambda actor, ref: loads.append(ref) or "goal text")
    assert loads == ["goal:1"]
    assert "secret:other" not in result.text and "secret:other" not in str(result.manifest)
    assert result.manifest["coverage"]["denied_count"] == 1


def test_required_sources_do_not_get_dropped_or_partially_clipped():
    refs = [EvidenceSource("goal:1", "goal"), EvidenceSource("receipt:1", "effect")]
    with pytest.raises(ContextBudgetExceeded):
        make_projection(refs, {"goal:1": "safe goal", "receipt:1": "z" * 1000}, budget=400)
    with pytest.raises(EvidenceAccessError, match="required_evidence_denied"):
        make_projection(refs, {"goal:1": "safe goal"})


def test_reauthorization_before_return_and_targeted_expansion():
    refs = [EvidenceSource("goal:1", "goal"), EvidenceSource("source:1", "observation")]
    data = {"goal:1": "goal", "source:1": "authorized detail"}
    access = {"goal:1", "source:1"}
    first = make_projection(refs, data, authorize=lambda actor, ref: ref in access)
    assert "authorized detail" in first.text
    access.remove("source:1")
    # Targeted expansion requires the source and rechecks current authority.
    with pytest.raises(EvidenceAccessError):
        make_projection([refs[0], EvidenceSource("source:1", "observation", required=True)], data,
                        authorize=lambda actor, ref: ref in access)

    checks = {}
    def revoked_while_loading(actor, ref):
        checks[ref] = checks.get(ref, 0) + 1
        return ref != "source:1" or checks[ref] == 1
    changed = make_projection(refs, data, authorize=revoked_while_loading)
    assert "authorized detail" not in changed.text
    assert changed.manifest["coverage"]["denied_count"] == 1


def test_producer_clipping_is_reported_and_oversized_optional_source_is_omitted():
    refs = [EvidenceSource("goal:1", "goal"), EvidenceSource("source:1", "observation", truncated=True),
            EvidenceSource("source:2", "observation")]
    result = make_projection(refs, {"goal:1": "goal", "source:1": "partial", "source:2": "x" * 5000}, budget=700)
    row = next(r for r in result.manifest["records"] if r["source_id"] == "source:1")
    assert row["truncated"] is True
    assert result.manifest["coverage"]["truncated_count"] == 1
    assert result.manifest["omitted_source_ids"] == ["source:2"]
    assert "x" * 100 not in result.text


def test_duplicate_reserved_or_missing_goal_rejected():
    with pytest.raises(ValueError, match="evidence_source_duplicate"):
        make_projection([EvidenceSource("goal:1", "goal"), EvidenceSource("goal:1", "observation")], {})
    with pytest.raises(ValueError, match="evidence_source_invalid"):
        make_projection([EvidenceSource("projection:coverage", "goal")], {})
    with pytest.raises(ValueError, match="evidence_goal_required"):
        make_projection([EvidenceSource("x", "observation")], {})


def test_required_clipped_source_fails_closed_and_optional_clipping_visible_in_text():
    refs = [EvidenceSource("goal:1", "goal"), EvidenceSource("receipt:1", "effect", truncated=True)]
    with pytest.raises(EvidenceAccessError, match="required_evidence_incomplete"):
        make_projection(refs, {"goal:1": "goal", "receipt:1": "partial receipt"})
    result = make_projection([refs[0], EvidenceSource("search:1", "observation", truncated=True)],
                             {"goal:1": "goal", "search:1": "partial"})
    assert "Producer marked this source as truncated" in result.text
    assert result.manifest["coverage"]["complete"] is False


def test_rechecks_earlier_source_after_loading_later_source():
    access = {"goal:1", "source:1", "source:2"}
    data = {"goal:1": "goal", "source:1": "private content", "source:2": "later content"}
    def load(actor, ref):
        if ref == "source:2":
            access.discard("source:1")
        return data[ref]
    result = project_evidence([EvidenceSource("goal:1", "goal"), EvidenceSource("source:1", "observation"),
                               EvidenceSource("source:2", "observation")], actor="alice", budget=1000,
                              authorize=lambda actor, ref: ref in access, load=load)
    assert "private content" not in result.text
    assert "source:1" not in str(result.manifest)
    assert result.manifest["coverage"]["complete"] is False


def test_optional_loader_error_is_a_private_explicit_omission():
    def load(actor, ref):
        if ref == "private:broken":
            raise RuntimeError("sensitive provider detail")
        return "goal"
    result = project_evidence([EvidenceSource("goal:1", "goal"),
                               EvidenceSource("private:broken", "observation")],
                              actor="alice", budget=1000,
                              authorize=lambda actor, ref: True, load=load)
    assert result.manifest["coverage"]["unavailable_count"] == 1
    assert result.manifest["coverage"]["omitted_count"] == 1
    assert result.manifest["coverage"]["complete"] is False
    assert "sensitive provider detail" not in str(result.manifest)
    assert "private:broken" not in str(result.manifest)


def test_provider_visible_coverage_names_omissions_without_denied_ids():
    refs = [EvidenceSource('goal:1', 'goal'), EvidenceSource('allowed:big', 'observation'),
            EvidenceSource('private:denied', 'observation'), EvidenceSource('clipped:1', 'observation', truncated=True)]
    data = {'goal:1': 'goal', 'allowed:big': 'x' * 9000, 'clipped:1': 'clipped'}
    result = make_projection(refs, data, budget=1000)
    assert 'allowed:big' in result.text and 'clipped:1' in result.text
    assert '"denied_count": 1' in result.text
    assert 'private:denied' not in result.text and 'private:denied' not in str(result.manifest)
    assert '"complete": false' in result.text
    assert len(result.text.encode('utf-8')) <= 1000


def test_coverage_that_cannot_fit_fails_closed():
    refs = [EvidenceSource('goal:1', 'goal')] + [EvidenceSource(f'source:{i}', 'observation') for i in range(200)]
    data = {'goal:1': 'goal', **{f'source:{i}': 'x' * 5000 for i in range(200)}}
    with pytest.raises(ContextBudgetExceeded):
        make_projection(refs, data, budget=600)
