from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

from sentinel.rag.parsing import LogRecord


@dataclass(frozen=True)
class LogStats:
    total: int
    by_level: dict[str, int]
    top_events: list[tuple[str, int]]
    first: datetime | None
    last: datetime | None
    problem_events: list[tuple[str, str, int]]


def compute_stats(
    records: Iterable[LogRecord],
    since: datetime | None = None,
    until: datetime | None = None,
    top: int = 8,
) -> LogStats:
    selected = [
        record
        for record in records
        if (since is None or record.timestamp >= since) and (until is None or record.timestamp < until)
    ]
    levels = Counter(record.level for record in selected)
    events = Counter(record.event for record in selected)
    problems = Counter(
        (record.level, record.event) for record in selected if record.level in ("warning", "error")
    )
    ordered_problems = sorted(problems.items(), key=lambda item: (item[0][0] != "error", -item[1], item[0][1]))
    return LogStats(
        total=len(selected),
        by_level=dict(levels),
        top_events=events.most_common(top),
        first=min((r.timestamp for r in selected), default=None),
        last=max((r.timestamp for r in selected), default=None),
        problem_events=[(level, event, count) for (level, event), count in ordered_problems],
    )


def render_stats(stats: LogStats) -> str:
    if stats.total == 0:
        return "no records in range\n"
    lines = [
        f"records: {stats.total}",
        f"range: {stats.first.isoformat(timespec='seconds')} to {stats.last.isoformat(timespec='seconds')}",
        "levels: " + ", ".join(f"{level}={count}" for level, count in sorted(stats.by_level.items())),
        "",
        "most frequent events:",
    ]
    lines.extend(f"  {event}: {count}" for event, count in stats.top_events)
    if stats.problem_events:
        lines.extend(["", "warnings and errors:"])
        lines.extend(f"  [{level}] {event}: {count}" for level, event, count in stats.problem_events)
    return "\n".join(lines) + "\n"
