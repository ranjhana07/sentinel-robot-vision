from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

from sentinel.config import Settings
from sentinel.rag.store import LogStore, RetrievedChunk

DEFAULT_QUESTION = (
    "Summarize notable incidents: errors, warnings, unknown people, restricted zone alerts, "
    "camera or audio failures and performance drops."
)

SYSTEM_PROMPT = (
    "You review logs from a surveillance robot for the operator. Use only the log excerpts provided. "
    "Group related lines into incidents, give the time range for each, say how severe it was and what "
    "the operator should check. If the excerpts show nothing notable, say so. Do not invent events."
)

HUMAN_PROMPT = "Time range: {since} to {until}\n\nLog excerpts:\n{context}\n\nTask: {question}"


@dataclass(frozen=True)
class Summary:
    text: str
    chunk_ids: tuple[str, ...]


def build_context(chunks: Sequence[RetrievedChunk], max_chars: int = 12000) -> str:
    sections: list[str] = []
    used = 0
    for chunk in chunks:
        section = f"[{chunk.start} to {chunk.end}]\n{chunk.text}"
        if used + len(section) > max_chars and sections:
            break
        sections.append(section)
        used += len(section) + 2
    return "\n\n".join(sections)


def build_llm(settings: Settings):
    from langchain.chat_models import init_chat_model

    return init_chat_model(
        settings.llm_model,
        model_provider=settings.llm_provider,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )


class LogSummarizer:
    def __init__(self, store: LogStore, llm, top_k: int = 6, max_context_chars: int = 12000) -> None:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        self._store = store
        self._top_k = top_k
        self._max_context_chars = max_context_chars
        prompt = ChatPromptTemplate.from_messages([("system", SYSTEM_PROMPT), ("human", HUMAN_PROMPT)])
        self._chain = prompt | llm | StrOutputParser()

    def retrieve(self, since: datetime, until: datetime, question: str) -> list[RetrievedChunk]:
        return self._store.query(question, since, until, self._top_k)

    def summarize(self, since: datetime, until: datetime, question: str | None = None) -> Summary:
        asked = question or DEFAULT_QUESTION
        chunks = self.retrieve(since, until, asked)
        if not chunks:
            return Summary("No log entries were found in this time range.", ())
        text = self._chain.invoke(
            {
                "since": since.isoformat(timespec="seconds"),
                "until": until.isoformat(timespec="seconds"),
                "context": build_context(chunks, self._max_context_chars),
                "question": asked,
            }
        )
        return Summary(text, tuple(chunk.chunk_id for chunk in chunks))
