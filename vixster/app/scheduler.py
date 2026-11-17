"""Daily refreshes while the server runs, so the data is current without clicking.

  * VX + FRED: weekdays after CBOE posts settlements (default 17:45 local)
  * Fund NAV (IBKR Flex): weekdays in the morning (default 08:30 local)

Runs through the same job runner as the buttons, so the page shows progress
and a click never doubles up with the scheduled run. Set VIXSTER_SCHEDULE=0
to turn it off.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timedelta

from vixster.app import jobs

log = logging.getLogger("vixster.scheduler")
_STARTED = False


def _times() -> list[tuple[str, str, object]]:
    from vixster.data import pipeline
    from vixster.perf import nav
    return [
        (os.getenv("VIXSTER_DATA_AT", "17:45"), "data", lambda lg: pipeline.run(log=lg)),
        (os.getenv("VIXSTER_NAV_AT", "08:30"), "nav", lambda lg: nav.pull_performance(log=lg)),
    ]


def _next(at: str, now: datetime) -> datetime:
    h, m = map(int, at.split(":"))
    t = now.replace(hour=h, minute=m, second=0, microsecond=0)
    while t <= now or t.weekday() >= 5:
        t += timedelta(days=1)
    return t


def _loop() -> None:
    plan = _times()
    due = {name: _next(at, datetime.now()) for at, name, _ in plan}
    while True:
        now = datetime.now()
        for at, name, fn in plan:
            if now >= due[name]:
                log.info("scheduled %s refresh", name)
                jobs.start(name, fn)
                due[name] = _next(at, now)
        time.sleep(30)


def start() -> None:
    global _STARTED
    if _STARTED or os.getenv("VIXSTER_SCHEDULE", "1") == "0":
        return
    _STARTED = True
    threading.Thread(target=_loop, daemon=True, name="scheduler").start()
