from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

from sentinel.config import RagConfig
from sentinel.rag.chunking import LogChunk


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    start: str
    end: str
    error_count: int
    warning_count: int
    distance: float | None = None


def window_filter(since: datetime, until: datetime) -> dict:
    return {
        "$and": [
            {"end_epoch": {"$gt": since.timestamp()}},
            {"start_epoch": {"$lt": until.timestamp()}},
        ]
    }


def to_retrieved(ids, documents, metadatas, distances=None) -> list[RetrievedChunk]:
    results = []
    for index, chunk_id in enumerate(ids):
        meta = metadatas[index]
        results.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                text=documents[index],
                start=meta["start"],
                end=meta["end"],
                error_count=int(meta["error_count"]),
                warning_count=int(meta["warning_count"]),
                distance=None if distances is None else float(distances[index]),
            )
        )
    return results


class LogStore:
    def __init__(
        self,
        persist_dir: Path | str,
        collection: str = "robot_logs",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection = client.get_or_create_collection(
            name=collection,
            embedding_function=SentenceTransformerEmbeddingFunction(model_name=embedding_model),
            metadata={"hnsw:space": "cosine"},
        )

    @classmethod
    def from_config(cls, config: RagConfig) -> "LogStore":
        return cls(config.persist_dir, config.collection, config.embedding_model)

    def add_chunks(self, chunks: Sequence[LogChunk]) -> int:
        if not chunks:
            return 0
        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[chunk.metadata() for chunk in chunks],
        )
        return len(chunks)

    def count(self) -> int:
        return self._collection.count()

    def query(self, text: str, since: datetime, until: datetime, top_k: int = 6) -> list[RetrievedChunk]:
        result = self._collection.query(
            query_texts=[text],
            n_results=top_k,
            where=window_filter(since, until),
        )
        found = to_retrieved(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
        return sorted(found, key=lambda chunk: chunk.start)

    def window(self, since: datetime, until: datetime) -> list[RetrievedChunk]:
        result = self._collection.get(where=window_filter(since, until))
        found = to_retrieved(result["ids"], result["documents"], result["metadatas"])
        return sorted(found, key=lambda chunk: chunk.start)
