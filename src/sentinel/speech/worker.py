import logging
import threading

from sentinel.speech.priority_queue import PhraseQueue
from sentinel.speech.tts import TtsBackend

logger = logging.getLogger(__name__)


class SpeechWorker:
    def __init__(self, backend: TtsBackend, max_queue: int = 8) -> None:
        self._backend = backend
        self._phrases = PhraseQueue(max_queue)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="speech", daemon=True)

    def start(self) -> "SpeechWorker":
        self._thread.start()
        return self

    def say(self, text: str, priority: int = 1) -> bool:
        return self._phrases.put(text, priority)

    def _run(self) -> None:
        while not self._stop.is_set() or len(self._phrases) > 0:
            text = self._phrases.get(timeout=0.1)
            if text is None:
                continue
            try:
                self._backend.speak(text)
            except Exception:
                logger.exception("speech backend failed")

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        self._thread.join(timeout)
        self._backend.close()
