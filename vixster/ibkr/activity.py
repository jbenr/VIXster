"""Plain-English IBKR status for the Data tab: one headline plus the recent steps.

Reads IBC's log (logs/gateway.out, fresh each Gateway start), the quote streamer's
log (logs/stream.log) and the clicks made on the page, and turns them into lines
like "Approval sent to your phone" or "Streaming 17 instruments".
"""
from __future__ import annotations

import re
import threading
from collections import deque
from datetime import datetime
from pathlib import Path

import pandas as pd

from vixster import config
from vixster.ibkr import gateway, procs

STREAM_LOG = config.LOG_DIR / "stream.log"
LIVE_SECONDS = 15          # quotes newer than this count as live

# (regex on an IBC log line, what to show, phase it puts Gateway in)
_IBC = [
    (r"IBC: Starting Gateway", "Starting IB Gateway", "starting"),
    (r"Login dialog WINDOW_OPENED", "Login window open", "login"),
    (r"Login attempt: (\d+)", "Entering username and password (attempt {0})", "login"),
    (r"Second Factor Authentication initiated", "Approval sent to your phone: open IBKR Mobile and approve", "2fa"),
    (r"Second Factor Authentication; event=Closed", "Phone approval received", "login"),
    (r"(?i)second factor.*(timed out|timeout)", "Phone approval timed out: sending a new one", "login"),
    (r"Login has completed", "Logged in to IBKR", "ready"),
    (r"Configuration tasks completed", "Gateway ready (API on port {port})", "ready"),
    (r"IBC returned exit status", "Gateway exited", "stopped"),
]
_STREAM = [
    (r"INFO connected to", "Quote streamer connected"),
    (r"INFO streaming (\d+) instruments", "Streaming {0} instruments"),
    (r"WARNING disconnected", "Quote streamer lost the connection"),
    (r"WARNING not connected", "Quote streamer waiting for Gateway"),
]

_CLICKS: deque = deque(maxlen=20)
_LOCK = threading.Lock()


def note(msg: str) -> None:
    """Record something the page did (a click and its result)."""
    with _LOCK:
        _CLICKS.append((datetime.now(), msg))


def _tail(path: Path, nbytes: int = 96_000) -> list[str]:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - nbytes))
            return f.read().decode(errors="ignore").splitlines()
    except OSError:
        return []


def _ibc_events() -> tuple[list[tuple[datetime, str]], str | None, datetime | None]:
    """Events from this Gateway run, the phase it's in, and when the phone request went out."""
    out, phase, sent = [], None, None
    try:   # for the few IBC lines without a timestamp (e.g. its exit status)
        mtime = datetime.fromtimestamp(gateway.GATEWAY_OUT.stat().st_mtime)
    except OSError:
        mtime = datetime.now()
    for line in _tail(gateway.GATEWAY_OUT, 400_000):
        m = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d):\d+ ", line)
        for rx, text, ph in _IBC:
            hit = re.search(rx, line)
            if not hit:
                continue
            when = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S") if m else mtime
            out.append((when, text.format(*hit.groups(), port=config.IB_PORT)))
            phase = ph
            if ph == "2fa":
                sent = when
            break
    return out, phase, sent


def _stream_events(since: datetime | None) -> list[tuple[datetime, str]]:
    out = []
    for line in _tail(STREAM_LOG):
        m = re.match(r"(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ ", line)
        if not m:
            continue
        when = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        if since and when < since:
            continue
        for rx, text in _STREAM:
            hit = re.search(rx, line)
            if hit:
                msg = text.format(*hit.groups())
                if not (out and out[-1][1] == msg):   # the retry line repeats every few seconds
                    out.append((when, msg))
                break
    return out


def _quote_age() -> float | None:
    try:
        q = pd.read_parquet(config.LIVE_QUOTES, columns=["Written"])
        return (pd.Timestamp.now() - pd.Timestamp(q["Written"].max())).total_seconds()
    except Exception:
        return None


def _ago(seconds: float) -> str:
    s = int(max(seconds, 0))
    return f"{s} s" if s < 60 else f"{s // 60}:{s % 60:02d}" if s < 3600 else f"{s // 3600} h {s % 3600 // 60} min"


def overview(limit: int = 8) -> dict:
    """{tone: up|warn|down, headline, button, events: [(time, text)]} for the page."""
    running = bool(gateway._pgrep("ibcalpha.ibc.IbcGateway"))
    api = gateway.port_open()
    streamer = procs.streamer_pid() is not None
    ibc, phase, sent = _ibc_events()
    first = ibc[0][0] if ibc else None
    with _LOCK:
        clicks = list(_CLICKS)
    events = sorted(ibc + _stream_events(first) + clicks, key=lambda e: e[0])[-limit:]
    age = _quote_age()

    if api:
        if age is not None and age < LIVE_SECONDS:
            tone, head = "up", f"Live: quotes updated {_ago(age)} ago"
        elif not streamer:
            tone, head = "warn", "Logged in, but the quote streamer isn't running: click IBKR Login to start it"
        elif age is None:
            tone, head = "warn", "Logged in: quote streamer connecting…"
        else:
            tone, head = "warn", f"Logged in, but no new quotes for {_ago(age)}"
    elif running:
        tone = "warn"
        if phase == "2fa":
            waited = f" (sent {_ago((datetime.now() - sent).total_seconds())} ago)" if sent else ""
            head = f"Waiting for you to approve the login in IBKR Mobile{waited}"
        elif phase in ("login", "ready"):
            head = "Logging in to IBKR…"
        else:
            head = "Starting IB Gateway…"
    else:
        tone, head = "down", "IBKR offline: click IBKR Login to connect"

    return {"tone": tone, "headline": head, "button": "Disconnect IBKR" if (running or api) else "IBKR Login",
            "events": events}
