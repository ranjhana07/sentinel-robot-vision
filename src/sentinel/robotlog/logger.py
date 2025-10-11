import json
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable

from sentinel.config import LoggingConfig


class RobotLogger:
    LEVELS = ("debug", "info", "warning", "error")

    def __init__(
        self,
        directory: Path | str,
        filename: str = "robot.jsonl",
        max_bytes: int = 5_000_000,
        backup_count: int = 5,
        robot_id: str = "sentinel-01",
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        folder = Path(directory)
        folder.mkdir(parents=True, exist_ok=True)
        self.path = folder / filename
        self._robot_id = robot_id
        self._now = now
        self._logger = logging.getLogger(f"sentinel.robotlog.{id(self)}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        self._handler = RotatingFileHandler(
            self.path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        self._handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(self._handler)

    @classmethod
    def from_config(cls, config: LoggingConfig) -> "RobotLogger":
        return cls(
            config.directory,
            config.filename,
            config.max_bytes,
            config.backup_count,
            config.robot_id,
        )

    def log(self, event: str, level: str = "info", **fields: Any) -> None:
        if level not in self.LEVELS:
            raise ValueError(f"unknown level {level!r}")
        record = {
            "ts": self._now().isoformat(timespec="milliseconds"),
            "level": level,
            "event": event,
            "robot": self._robot_id,
            **fields,
        }
        self._logger.info(json.dumps(record, default=str, separators=(",", ":")))

    def close(self) -> None:
        self._logger.removeHandler(self._handler)
        self._handler.close()
