"""Tiny background-job runner: slow buttons start a thread, the page polls its log."""
from __future__ import annotations

import threading
import time
import traceback
from typing import Callable

_JOBS: dict[str, dict] = {}
_LOCK = threading.Lock()


def start(name: str, fn: Callable[[Callable[[str], None]], object]) -> bool:
    """Run fn(log) in a thread unless a job with this name is already running."""
    with _LOCK:
        if _JOBS.get(name, {}).get("running"):
            return False
        job = {"running": True, "lines": [], "ok": None, "result": None, "error": None,
               "started": time.time(), "finished": None}
        _JOBS[name] = job

    def log(msg: str) -> None:
        job["lines"].append(f"{time.strftime('%H:%M:%S')}  {msg}")

    def run():
        try:
            job["result"] = fn(log)
            job["ok"] = True
        except Exception as e:
            job["error"] = str(e)
            log(f"ERROR: {e}")
            log(traceback.format_exc(limit=2).strip().splitlines()[-1])
            job["ok"] = False
        finally:
            job["running"] = False
            job["finished"] = time.time()

    threading.Thread(target=run, daemon=True, name=f"job-{name}").start()
    return True


def get(name: str) -> dict:
    job = _JOBS.get(name, {"running": False, "lines": [], "ok": None, "result": None, "error": None,
                           "finished": None})
    return dict(job, lines=list(job["lines"]))


def messages(job: dict) -> list[str]:
    """The log lines without their timestamps, as the page shows them."""
    return [line[10:] for line in job["lines"]]
