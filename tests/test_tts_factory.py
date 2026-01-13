import pytest

from sentinel.config import SpeechConfig
from sentinel.speech.factory import build_backend
from sentinel.speech.tts import ConsoleBackend, Pyttsx3Backend


def test_console_backend_prints(capsys):
    backend = build_backend(SpeechConfig(backend="console"))
    assert isinstance(backend, ConsoleBackend)
    backend.speak("hello")
    assert "hello" in capsys.readouterr().out


def test_pyttsx3_backend_is_lazy():
    backend = build_backend(SpeechConfig(backend="pyttsx3", rate=150))
    assert isinstance(backend, Pyttsx3Backend)
    backend.close()


def test_piper_needs_a_model_path():
    with pytest.raises(ValueError):
        build_backend(SpeechConfig(backend="piper"))


def test_piper_reports_missing_binary(tmp_path):
    config = SpeechConfig(backend="piper", piper_model=tmp_path / "voice.onnx", piper_binary="definitely-not-installed")
    with pytest.raises(FileNotFoundError):
        build_backend(config)
