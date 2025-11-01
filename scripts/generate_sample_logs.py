import argparse
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

DAY = datetime(2025, 11, 4, 8, 0, 0, tzinfo=timezone.utc)
KNOWN_NAMES = ["Asha", "Ravi", "Meena"]


def at(hour: int, minute: int, second: int = 0) -> datetime:
    return DAY.replace(hour=hour, minute=minute, second=second)


def record(stamp: datetime, level: str, event: str, **fields) -> dict:
    return {
        "ts": stamp.isoformat(timespec="milliseconds"),
        "level": level,
        "event": event,
        "robot": "sentinel-01",
        **fields,
    }


def routine(rng: random.Random) -> list[dict]:
    records = [
        record(at(8, 0, 2), "info", "startup", model="face-yolov8n-int8.onnx"),
        record(at(8, 0, 5), "info", "camera_open", source=0, width=640, height=480),
    ]
    stamp = at(8, 5)
    while stamp < at(18, 0):
        records.append(record(stamp, "info", "fps", fps=round(rng.uniform(9.2, 10.4), 2)))
        stamp += timedelta(minutes=5)
    track = 1
    stamp = at(8, 10)
    while stamp < at(17, 50):
        stamp += timedelta(minutes=rng.randint(4, 14), seconds=rng.randint(0, 59))
        known = rng.random() < 0.6
        name = rng.choice(KNOWN_NAMES) if known else "unknown"
        records.append(
            record(stamp, "info", "face_detected", track=track, identity=name, score=round(rng.uniform(0.55, 0.93), 3), zone="entrance")
        )
        text = f"Hello {name}" if known else "Unknown person detected"
        records.append(
            record(stamp + timedelta(seconds=1), "info", "speech", kind="greeting" if known else "unknown", text=text, queued=True)
        )
        records.append(record(stamp + timedelta(seconds=rng.randint(8, 40)), "info", "face_lost", track=track))
        track += 1
    return records


def incidents() -> list[dict]:
    records = []
    for offset, fps in enumerate([6.1, 4.8, 4.1, 4.4, 7.9]):
        level = "warning" if fps < 8 else "info"
        event = "fps_drop" if fps < 8 else "fps_recovered"
        records.append(record(at(9, 47, 10 + offset * 20), level, event, fps=fps, threshold=8))
    for offset in range(3):
        records.append(
            record(at(10, 15, 3 + offset * 6), "error", "tts_failure", backend="pyttsx3", error="audio device busy")
        )
    records.append(record(at(11, 30, 20), "info", "face_detected", track=90, identity="unknown", score=0.81, zone="server_room"))
    records.append(record(at(11, 30, 21), "warning", "zone_violation", zone="server_room", track=90))
    records.append(record(at(11, 30, 22), "info", "speech", kind="zone:restricted", text="This area is restricted. Please step back.", queued=True))
    records.append(record(at(11, 31, 2), "info", "face_lost", track=90))
    for attempt in range(1, 4):
        records.append(record(at(13, 5, 44 + attempt), "error", "camera_read_failed", attempt=attempt, device="/dev/video0"))
    records.append(record(at(13, 5, 52), "warning", "camera_reconnect", device="/dev/video0"))
    records.append(record(at(13, 5, 55), "info", "camera_open", source=0, width=640, height=480))
    records.append(record(at(15, 40, 12), "warning", "disk_low", free_mb=210, path="/var/log/sentinel"))
    for track, second in ((95, 5), (96, 9), (97, 14)):
        records.append(record(at(16, 20, second), "info", "face_detected", track=track, identity="unknown", score=0.74, zone="entrance"))
    records.append(record(at(16, 20, 16), "info", "speech", kind="unknown", text="3 unknown people detected", queued=True))
    for offset, temp in enumerate([79.0, 82.5, 83.1]):
        records.append(record(at(17, 5, offset * 30), "warning", "thermal_throttle", cpu_temp_c=temp))
    records.append(record(at(18, 0, 1), "info", "shutdown", frames=351204))
    return records


def build(seed: int) -> list[dict]:
    rng = random.Random(seed)
    return sorted(routine(rng) + incidents(), key=lambda item: item["ts"])


def main() -> None:
    parser = argparse.ArgumentParser(description="write a synthetic robot log with known incidents")
    parser.add_argument("--output", type=Path, default=Path("examples/sample_robot.jsonl"))
    parser.add_argument("--seed", type=int, default=11)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        for item in build(args.seed):
            handle.write(json.dumps(item, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
