from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DetectorConfig(BaseModel):
    model_path: Path = Path("models/face-yolov8n.onnx")
    input_size: int = 416
    conf_threshold: float = Field(0.35, ge=0.0, le=1.0)
    iou_threshold: float = Field(0.45, ge=0.0, le=1.0)
    providers: list[str] = ["CPUExecutionProvider"]
    intra_op_threads: int = Field(4, ge=1)


class SourceConfig(BaseModel):
    kind: Literal["camera", "video", "picamera"] = "camera"
    target: int | str = 0
    width: int = 640
    height: int = 480
    queue_size: int = Field(2, ge=1)


class TrackerConfig(BaseModel):
    iou_threshold: float = Field(0.3, ge=0.0, le=1.0)
    max_age: int = Field(15, ge=0)
    min_hits: int = Field(2, ge=1)
    use_motion: bool = True
    velocity_smoothing: float = Field(0.6, ge=0.0, lt=1.0)


class ZoneRule(BaseModel):
    name: str
    area: tuple[float, float, float, float]
    phrase: str
    applies_to: Literal["any", "known", "unknown"] = "any"
    cooldown_seconds: float | None = None

    @field_validator("area")
    @classmethod
    def area_must_be_normalised(cls, value: tuple[float, float, float, float]):
        x1, y1, x2, y2 = value
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            raise ValueError("zone area must be normalised (x1, y1, x2, y2) with x1 < x2 and y1 < y2")
        return value


class SpeechConfig(BaseModel):
    enabled: bool = True
    backend: Literal["pyttsx3", "piper", "console"] = "pyttsx3"
    rate: int = 170
    volume: float = Field(1.0, ge=0.0, le=1.0)
    piper_binary: str = "piper"
    piper_model: Path | None = None
    queue_size: int = Field(8, ge=1)
    cooldown_seconds: float = 20.0
    unknown_cooldown_seconds: float = 45.0
    greeting: str = "Hello {name}"
    unknown_phrase: str = "Unknown person detected"
    group_phrase: str = "{count} unknown people detected"
    zones: list[ZoneRule] = []


class LoggingConfig(BaseModel):
    directory: Path = Path("logs")
    filename: str = "robot.jsonl"
    max_bytes: int = 5_000_000
    backup_count: int = 5
    robot_id: str = "sentinel-01"


class RagConfig(BaseModel):
    persist_dir: Path = Path("data/chroma")
    collection: str = "robot_logs"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    window_minutes: int = Field(10, ge=1)
    max_chunk_chars: int = Field(3000, ge=200)
    top_k: int = Field(6, ge=1)


class AppConfig(BaseModel):
    detector: DetectorConfig = DetectorConfig()
    source: SourceConfig = SourceConfig()
    tracker: TrackerConfig = TrackerConfig()
    speech: SpeechConfig = SpeechConfig()
    rag: RagConfig = RagConfig()
    logging: LoggingConfig = LoggingConfig()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SENTINEL_", env_file=".env", extra="ignore")

    config_path: Path = Path("configs/default.yaml")
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-5-5"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1024


def load_config(path: Path | str) -> AppConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return AppConfig.model_validate(raw)
