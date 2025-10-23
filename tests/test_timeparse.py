from datetime import datetime, timedelta, timezone

import pytest

from sentinel.rag.timeparse import parse_time_argument

NOW = datetime(2025, 11, 4, 12, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    "value,delta",
    [("30m", timedelta(minutes=30)), ("2h", timedelta(hours=2)), ("1d", timedelta(days=1)), ("45 s", timedelta(seconds=45))],
)
def test_relative_values(value, delta):
    assert parse_time_argument(value, NOW) == NOW - delta


def test_now_keyword():
    assert parse_time_argument("now", NOW) == NOW


def test_iso_values():
    assert parse_time_argument("2025-11-04T09:00:00Z", NOW) == datetime(2025, 11, 4, 9, tzinfo=timezone.utc)
    assert parse_time_argument("2025-11-04", NOW) == datetime(2025, 11, 4, tzinfo=timezone.utc)


def test_bad_value_raises_helpful_error():
    with pytest.raises(ValueError, match="relative value"):
        parse_time_argument("yesterday-ish", NOW)
