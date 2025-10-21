from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from itertools import groupby
from typing import Iterable

from sentinel.rag.parsing import LogRecord

SEVERITY = {"debug": 0, "info": 1, "warning": 2, "error": 3}


@dataclass(frozen=True)
class LogChunk:
    chunk_id: str
    start: datetime
    end: datetime
    text: str
    record_count: int
    error_count: int
    warning_count: int

    def metadata(self) -> dict[str, str | int | float]:
        return {
            "start_epoch": self.start.timestamp(),
            "end_epoch": self.end.timestamp(),
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "record_count": self.record_count,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
        }


def format_record(record: LogRecord) -> str:
    stamp = record.timestamp.strftime("%Y-%m-%d %H:%M:%S")
    return f"{stamp} {record.level.upper()} {record.message}"


def window_start(timestamp: datetime, window: timedelta) -> datetime:
    seconds = int(window.total_seconds())
    bucket = int(timestamp.timestamp()) // seconds
    return datetime.fromtimestamp(bucket * seconds, tz=timezone.utc)


def split_lines(lines: list[str], max_chars: int) -> list[list[str]]:
    parts: list[list[str]] = [[]]
    size = 0
    for line in lines:
        if parts[-1] and size + len(line) + 1 > max_chars:
            parts.append([])
            size = 0
        parts[-1].append(line)
        size += len(line) + 1
    return parts


def chunk_by_window(
    records: Iterable[LogRecord],
    window: timedelta = timedelta(minutes=10),
    max_chars: int = 3000,
) -> list[LogChunk]:
    ordered = sorted(records, key=lambda record: record.timestamp)
    chunks: list[LogChunk] = []
    for start, group in groupby(ordered, key=lambda record: window_start(record.timestamp, window)):
        members = list(group)
        for part, lines in enumerate(split_lines([format_record(r) for r in members], max_chars)):
            chunks.append(
                LogChunk(
                    chunk_id=f"{start:%Y%m%dT%H%M%S}-{part}",
                    start=start,
                    end=start + window,
                    text="\n".join(lines),
                    record_count=len(lines),
                    error_count=sum(1 for r in members if r.level == "error"),
                    warning_count=sum(1 for r in members if r.level == "warning"),
                )
            )
    return chunks
