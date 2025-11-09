import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

TEXT_LINE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?)"
    r"\s+(?P<level>[A-Za-z]+)\s+(?P<message>.*)$"
)
RESERVED_KEYS = {"ts", "level", "event", "robot"}
LEVEL_ALIASES = {"warn": "warning", "err": "error", "critical": "error", "fatal": "error"}


@dataclass(frozen=True)
class LogRecord:
    timestamp: datetime
    level: str
    event: str
    message: str


def parse_timestamp(value: str) -> datetime:
    cleaned = value.strip().replace(",", ".").replace(" ", "T", 1)
    if cleaned.endswith("Z"):
        cleaned = cleaned[:-1] + "+00:00"
    parsed = datetime.fromisoformat(cleaned)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def normalise_level(level: str) -> str:
    lowered = level.lower()
    return LEVEL_ALIASES.get(lowered, lowered)


def parse_json_line(line: str) -> LogRecord | None:
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or "ts" not in data:
        return None
    try:
        timestamp = parse_timestamp(str(data["ts"]))
    except ValueError:
        return None
    event = str(data.get("event", "log"))
    extras = " ".join(f"{key}={value}" for key, value in data.items() if key not in RESERVED_KEYS)
    message = f"{event} {extras}".strip()
    return LogRecord(timestamp, normalise_level(str(data.get("level", "info"))), event, message)


def parse_text_line(line: str) -> LogRecord | None:
    match = TEXT_LINE.match(line.strip())
    if match is None:
        return None
    try:
        timestamp = parse_timestamp(match["ts"])
    except ValueError:
        return None
    message = match["message"].strip()
    return LogRecord(timestamp, normalise_level(match["level"]), message.split(" ", 1)[0], message)


def parse_line(line: str) -> LogRecord | None:
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith("{"):
        return parse_json_line(stripped)
    return parse_text_line(stripped)


def read_records(path: Path | str) -> Iterator[LogRecord]:
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            record = parse_line(line)
            if record is not None:
                yield record
