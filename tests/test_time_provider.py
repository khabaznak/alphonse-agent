from core.context.providers.time_provider import get_day_period


def test_get_day_period_morning():
    assert get_day_period(6) == "morning"
    assert get_day_period(11) == "morning"


def test_get_day_period_afternoon():
    assert get_day_period(12) == "afternoon"
    assert get_day_period(17) == "afternoon"


def test_get_day_period_evening():
    assert get_day_period(18) == "evening"
    assert get_day_period(21) == "evening"


def test_get_day_period_night():
    assert get_day_period(0) == "night"
    assert get_day_period(5) == "night"
    assert get_day_period(22) == "night"
    assert get_day_period(23) == "night"
