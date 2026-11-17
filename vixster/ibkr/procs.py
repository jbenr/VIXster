"""Start/stop the quote streamer as its own process (survives app restarts)."""
from __future__ import annotations

import os
import signal
import subprocess
import sys

from vixster import config
from vixster.ibkr.stream import PIDFILE


def streamer_pid() -> int | None:
    try:
        pid = int(PIDFILE.read_text())
        os.kill(pid, 0)
        return pid
    except (FileNotFoundError, ValueError, ProcessLookupError, PermissionError):
        return None


def start_streamer() -> str:
    if streamer_pid():
        return "Streamer already running."
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.LOG_DIR / "stream.out", "ab") as out:
        subprocess.Popen([sys.executable, "-m", "vixster.ibkr.stream"], stdout=out, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True, cwd=str(config.ROOT))
    return "Streamer started."


def stop_streamer() -> str:
    pid = streamer_pid()
    if not pid:
        return "Streamer was not running."
    os.kill(pid, signal.SIGTERM)
    return "Streamer stopped."
