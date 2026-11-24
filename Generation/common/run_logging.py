"""Tees stdout/stderr to a per-run log file under Generation/logs/, on top
of whatever the terminal already shows."""
import sys
import time
from pathlib import Path

LOGS_DIR = Path(__file__).resolve().parent.parent / "logs"


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def setup_logging(script_name: str) -> Path:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_DIR / f"{script_name}_{timestamp}.log"
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)

    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)

    print(f"[Logging] {script_name} -- writing to {log_path}")
    return log_path
