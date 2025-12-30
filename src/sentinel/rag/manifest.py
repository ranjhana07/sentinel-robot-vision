import hashlib
import json
from pathlib import Path


def file_digest(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class IngestManifest:
    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._entries: dict[str, str] = {}
        if self._path.exists():
            try:
                self._entries = json.loads(self._path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self._entries = {}

    def is_current(self, log_path: Path | str) -> bool:
        key = str(Path(log_path).resolve())
        return self._entries.get(key) == file_digest(log_path)

    def record(self, log_path: Path | str) -> None:
        self._entries[str(Path(log_path).resolve())] = file_digest(log_path)

    def save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._entries, indent=2, sort_keys=True), encoding="utf-8")

    def __len__(self) -> int:
        return len(self._entries)
