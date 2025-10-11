import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class TtsBackend(Protocol):
    def speak(self, text: str) -> None: ...

    def close(self) -> None: ...


class ConsoleBackend:
    def speak(self, text: str) -> None:
        print(f"[speech] {text}", flush=True)

    def close(self) -> None:
        return None


class Pyttsx3Backend:
    def __init__(self, rate: int = 170, volume: float = 1.0) -> None:
        self._rate = rate
        self._volume = volume
        self._engine = None

    def _ensure_engine(self):
        if self._engine is None:
            import pyttsx3

            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self._rate)
            self._engine.setProperty("volume", self._volume)
        return self._engine

    def speak(self, text: str) -> None:
        engine = self._ensure_engine()
        engine.say(text)
        engine.runAndWait()

    def close(self) -> None:
        if self._engine is not None:
            self._engine.stop()
            self._engine = None
