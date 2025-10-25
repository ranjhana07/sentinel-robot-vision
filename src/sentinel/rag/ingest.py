from datetime import timedelta
from pathlib import Path
from typing import Iterable

from sentinel.rag.chunking import LogChunk, chunk_by_window
from sentinel.rag.parsing import LogRecord, read_records


def expand_paths(paths: Iterable[Path | str]) -> list[Path]:
    files: list[Path] = []
    for entry in paths:
        path = Path(entry)
        if path.is_dir():
            files.extend(sorted(p for p in path.iterdir() if p.is_file()))
        else:
            files.append(path)
    return files


def load_records(paths: Iterable[Path | str]) -> list[LogRecord]:
    records: list[LogRecord] = []
    for path in expand_paths(paths):
        records.extend(read_records(path))
    return records


def build_chunks(
    paths: Iterable[Path | str],
    window_minutes: int = 10,
    max_chars: int = 3000,
) -> list[LogChunk]:
    return chunk_by_window(load_records(paths), timedelta(minutes=window_minutes), max_chars)
