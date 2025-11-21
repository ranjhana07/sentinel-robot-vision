import os
import platform


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.lower().startswith(("model name", "hardware", "model\t")):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    if platform.system() == "Windows":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            )
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    return platform.processor() or platform.machine()


def environment_info() -> dict[str, str | int]:
    import onnxruntime

    info: dict[str, str | int] = {
        "cpu": cpu_name(),
        "logical_cores": os.cpu_count() or 0,
        "os": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "python": platform.python_version(),
        "onnxruntime": onnxruntime.__version__,
    }
    try:
        import torch

        info["torch"] = torch.__version__
    except ImportError:
        pass
    return info
