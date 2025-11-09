import re
from datetime import datetime, timedelta, timezone

from sentinel.rag.parsing import parse_timestamp

RELATIVE = re.compile(r"^(?P<amount>\d+)\s*(?P<unit>[smhdw])$")
UNIT_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def parse_time_argument(value: str, now: datetime | None = None) -> datetime:
    reference = now or datetime.now(timezone.utc)
    text = value.strip().lower()
    if text == "now":
        return reference
    match = RELATIVE.match(text)
    if match:
        seconds = int(match["amount"]) * UNIT_SECONDS[match["unit"]]
        return reference - timedelta(seconds=seconds)
    try:
        return parse_timestamp(value)
    except ValueError as error:
        raise ValueError(f"cannot parse time {value!r}, use ISO format or a relative value like 2h") from error
