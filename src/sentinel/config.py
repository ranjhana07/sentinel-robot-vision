from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
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


class LoggingConfig(BaseModel):
    directory: Path = Path("logs")
    filename: str = "robot.jsonl"
    max_bytes: int = 5_000_000
    backup_count: int = 5
    robot_id: str = "sentinel-01"


class AppConfig(BaseModel):
    detector: DetectorConfig = DetectorConfig()
    source: SourceConfig = SourceConfig()
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
