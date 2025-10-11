from sentinel.config import SpeechConfig
from sentinel.speech.tts import ConsoleBackend, Pyttsx3Backend, TtsBackend


def build_backend(config: SpeechConfig) -> TtsBackend:
    if config.backend == "console":
        return ConsoleBackend()
    if config.backend == "piper":
        from sentinel.speech.piper import PiperBackend

        if config.piper_model is None:
            raise ValueError("speech.piper_model is required for the piper backend")
        return PiperBackend(config.piper_model, config.piper_binary)
    return Pyttsx3Backend(config.rate, config.volume)
