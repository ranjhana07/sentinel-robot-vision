import shutil
import subprocess
from pathlib import Path

DEFAULT_PLAYER = ("aplay", "-q", "-r", "22050", "-f", "S16_LE", "-t", "raw", "-")


class PiperBackend:
    def __init__(
        self,
        model_path: Path | str,
        binary: str = "piper",
        player: tuple[str, ...] = DEFAULT_PLAYER,
    ) -> None:
        if shutil.which(binary) is None:
            raise FileNotFoundError(f"piper binary {binary!r} not found on PATH")
        self._command = [binary, "--model", str(model_path), "--output_raw"]
        self._player = list(player)

    def speak(self, text: str) -> None:
        synth = subprocess.Popen(self._command, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        player = subprocess.Popen(self._player, stdin=synth.stdout)
        assert synth.stdin is not None and synth.stdout is not None
        synth.stdin.write(text.encode("utf-8"))
        synth.stdin.close()
        synth.stdout.close()
        player.wait()
        synth.wait()

    def close(self) -> None:
        return None
