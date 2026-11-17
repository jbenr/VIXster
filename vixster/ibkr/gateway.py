"""Start, stop and check IB Gateway, logged in automatically by IBC.

Why the old login was flaky:
  * It typed the username/password into the window with xdotool after a fixed
    `sleep 5`. If the window wasn't up yet, or the 2FA prompt appeared, the
    keystrokes went nowhere.
  * The earlier IBC attempt ran Gateway on the desktop display (:1); when that
    X connection dropped, Java crashed (see ~/Jts/ibgateway/1030/hs_err_*.log).
  * It set IBC's CommandServerPort to 4001, the same port Gateway's API uses.

This version:
  * Runs IBC 3.24 (handles the login dialog, waits for the IBKR Mobile 2FA
    approval, retries if the 2FA prompt times out, and restarts Gateway every
    night so the session survives the week without another 2FA).
  * Gives Gateway its own virtual display (Xvfb :99), started if missing.
  * Writes a private IBC config (data/ibc/config.ini, mode 600) from
    IB_USER / IB_PASS in .env, so credentials never appear on a command line
    or in IBC's log.
  * Talks to IBC's command server on 7462 for a clean stop.
"""
from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from vixster import config

IBC_DIR = Path(os.getenv("IBC_NEW_PATH", Path.home() / "ibc-3.24.2"))
IBC_INI = config.DATA / "ibc" / "config.ini"
COMMAND_PORT = int(os.getenv("IBC_COMMAND_PORT", "7462"))
AUTO_RESTART = os.getenv("IB_AUTO_RESTART", "11:55 PM")


def _ini_text(user: str, password: str) -> str:
    settings = {
        "IbLoginId": user,
        "IbPassword": password,
        "TradingMode": config.IB_TRADING_MODE,
        "FIX": "no",
        # 2FA: wait up to 3 min for the phone approval, then retry the login
        "SecondFactorAuthenticationTimeout": "180",
        "ReloginAfterSecondFactorAuthenticationTimeout": "yes",
        "ExitAfterSecondFactorAuthenticationTimeout": "no",
        "AcceptNonBrokerageAccountWarning": "yes",
        "ExistingSessionDetectedAction": "primary",
        "OverrideTwsApiPort": str(config.IB_PORT),
        "ReadOnlyApi": os.getenv("IB_READ_ONLY_API", "yes"),   # quotes only; flip to "no" to trade via API
        "AcceptIncomingConnectionAction": "reject",            # localhost is trusted already
        "TrustedTwsApiClientIPs": "127.0.0.1",
        "AutoRestartTime": AUTO_RESTART,                       # nightly soft restart keeps the session alive
        "StoreSettingsOnServer": "no",
        "MinimizeMainWindow": "no",
        "AllowBlindTrading": "no",
        "DismissPasswordExpiryWarning": "no",
        "DismissNSEComplianceNotice": "yes",
        "CommandServerPort": str(COMMAND_PORT),
        "ControlFrom": "127.0.0.1",
        "BindAddress": "127.0.0.1",
        "SuppressInfoMessages": "yes",
        "LogStructureScope": "known",
        "LogStructureWhen": "never",
    }
    return "\n".join(f"{k}={v}" for k, v in settings.items()) + "\n"


def write_ini() -> Path:
    user, pw = os.getenv("IB_USER"), os.getenv("IB_PASS")
    if not user or not pw:
        raise RuntimeError("IB_USER / IB_PASS are not set (VIXster/.env)")
    IBC_INI.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(IBC_INI, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(_ini_text(user, pw))
    os.chmod(IBC_INI, 0o600)
    return IBC_INI


# ── status ───────────────────────────────────────────────────────────────────

def port_open(host: str = config.IB_HOST, port: int = config.IB_PORT, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _pgrep(pattern: str) -> list[int]:
    r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    return [int(p) for p in r.stdout.split()]


GATEWAY_OUT = config.LOG_DIR / "gateway.out"   # IBC's own log (stdout of ibcstart.sh)


def _latest_ibc_log() -> Path | None:
    return GATEWAY_OUT if GATEWAY_OUT.exists() else None


_PHASES = [  # (regex on IBC log, phase) — the last match wins
    (r"Starting Gateway|Starting TWS", "starting"),
    (r"Login dialog|detected frame entitled: (IBKR|IB) Gateway", "login"),
    (r"Second Factor Authentication|2FA|IBKR Mobile", "2fa"),
    (r"Login has completed|Configuration tasks completed", "ready"),
    (r"re-login|Relogin|relogin", "retry"),
    (r"IBC closing|Shutdown|shutting down|exiting", "stopped"),
]


def _log_phase(log: Path | None) -> tuple[str | None, str]:
    if not log:
        return None, ""
    lines = log.read_text(errors="ignore").splitlines()[-400:]
    phase, last = None, ""
    for line in lines:
        for rx, ph in _PHASES:
            if re.search(rx, line, re.I):
                phase, last = ph, line
    return phase, re.sub(r"^\S+ \S+ IBC: ", "", last)[:160]


@dataclass
class Status:
    api_up: bool
    running: bool
    phase: str
    detail: str

    @property
    def label(self) -> str:
        if self.api_up:
            return "Connected"
        return {"2fa": "Approve on phone", "login": "Logging in", "starting": "Starting",
                "retry": "Retrying login", "ready": "Starting API"}.get(self.phase, "Offline")

    @property
    def tone(self) -> str:
        return "up" if self.api_up else ("warn" if self.running else "down")


def status() -> Status:
    api = port_open()
    running = bool(_pgrep("ibcalpha.ibc.IbcGateway"))
    phase, detail = _log_phase(_latest_ibc_log())
    if api:
        phase = "ready"
    elif not running:
        phase = "stopped"
    return Status(api, running, phase or "stopped", detail)


# ── start / stop ─────────────────────────────────────────────────────────────

def _ensure_display() -> str:
    disp = config.IB_DISPLAY
    n = disp.lstrip(":")
    if Path(f"/tmp/.X11-unix/X{n}").exists() and _pgrep(f"Xvfb {disp}"):
        return disp
    Path(f"/tmp/.X{n}-lock").unlink(missing_ok=True)
    if not shutil.which("Xvfb"):
        raise RuntimeError("Xvfb is not installed (sudo apt install xvfb)")
    subprocess.Popen(["Xvfb", disp, "-screen", "0", "1280x1024x24", "-nolisten", "tcp"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    for _ in range(30):
        if Path(f"/tmp/.X11-unix/X{n}").exists():
            return disp
        time.sleep(0.1)
    raise RuntimeError(f"Xvfb {disp} did not start")


def start() -> str:
    """Launch Gateway via IBC in the background. Returns a message for the UI.
    Approve the IBKR Mobile notification when it arrives."""
    st = status()
    if st.api_up:
        return "Gateway is already connected."
    if st.running:
        return f"Gateway is already starting ({st.label.lower()})."
    if not (IBC_DIR / "scripts" / "ibcstart.sh").exists():
        raise RuntimeError(f"IBC not found at {IBC_DIR}")
    ini = write_ini()
    env = {k: v for k, v in os.environ.items() if k not in ("IB_USER", "IB_PASS", "JAVA_TOOL_OPTIONS")}
    env.update(DISPLAY=_ensure_display())
    cmd = [str(IBC_DIR / "scripts" / "ibcstart.sh"), config.TWS_MAJOR_VRSN, "--gateway",
           f"--tws-path={config.TWS_PATH}", f"--ibc-path={IBC_DIR}", f"--ibc-ini={ini}",
           f"--mode={config.IB_TRADING_MODE}", "--on2fatimeout=restart"]
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(GATEWAY_OUT, "wb") as out:  # fresh log per start, so status reflects this run
        subprocess.Popen(cmd, env=env, stdout=out, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True, cwd=str(IBC_DIR))
    return "Gateway starting — approve the IBKR Mobile notification on your phone."


def _command(cmd: str) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", COMMAND_PORT), timeout=2) as s:
            s.sendall(f"{cmd}\n".encode())
            s.recv(256)
        return True
    except OSError:
        return False


def stop() -> str:
    if _command("STOP"):
        return "Stop sent to Gateway."
    pids = _pgrep("ibcalpha.ibc.IbcGateway")
    for p in pids:
        try:
            os.kill(p, 15)
        except ProcessLookupError:
            pass
    return "Gateway stopped." if pids else "Gateway was not running."


def restart() -> str:
    stop()
    for _ in range(40):
        if not status().running:
            break
        time.sleep(0.5)
    return start()


if __name__ == "__main__":
    import sys
    action = sys.argv[1] if len(sys.argv) > 1 else "status"
    if action == "status":
        print(status())
    else:
        print({"start": start, "stop": stop, "restart": restart}[action]())
