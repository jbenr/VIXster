"""`vixster <command>` — everything you'd run by hand.

  vixster serve            production server (gunicorn) on localhost + Tailscale
  vixster app              dev server with auto-reload
  vixster data [--backfill]
  vixster nav              pull IBKR Flex NAV
  vixster signal           print today's signal in the terminal
  vixster retrain [--apply] re-run the all-subsets feature search (~20s) and refit
  vixster stream           run the IBKR quote streamer in the foreground
  vixster gateway start|stop|restart|status
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


def _tailscale_ip() -> str | None:
    try:
        out = subprocess.run(["tailscale", "ip", "-4"], capture_output=True, text=True, timeout=5).stdout.split()
        return out[0] if out else None
    except Exception:
        return None


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="vixster", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=int(os.getenv("VIXSTER_PORT", "8050")))
    a = sub.add_parser("app")
    a.add_argument("--port", type=int, default=8050)
    d = sub.add_parser("data")
    d.add_argument("--backfill", action="store_true")
    sub.add_parser("nav")
    sub.add_parser("signal")
    r = sub.add_parser("retrain")
    r.add_argument("--apply", action="store_true", help="make the new models live")
    sub.add_parser("stream")
    g = sub.add_parser("gateway")
    g.add_argument("action", choices=["start", "stop", "restart", "status"])
    args = ap.parse_args(argv)

    if args.cmd == "serve":
        binds = ["127.0.0.1"] + ([ip] if (ip := _tailscale_ip()) else [])
        cmd = [sys.executable, "-m", "gunicorn", "vixster.app.wsgi:server", "--workers", "1", "--threads", "8",
               "--timeout", "180", "--access-logfile", "-"]
        for b in binds:
            cmd += ["--bind", f"{b}:{args.port}"]
        print("Serving on " + ", ".join(f"http://{b}:{args.port}" for b in binds), flush=True)
        os.execv(sys.executable, cmd)
    elif args.cmd == "app":
        from vixster.app.app import app
        app.run(host="127.0.0.1", port=args.port, debug=True)
    elif args.cmd == "data":
        from vixster.data import pipeline
        pipeline.run(backfill=args.backfill)
    elif args.cmd == "nav":
        from vixster.perf import nav
        nav.pull_performance()
    elif args.cmd == "signal":
        import pandas as pd
        from vixster.model import service
        sig = service.live_signal(use_live=True)
        pd.set_option("display.width", 160)
        print(f"{sig['meta']['mode']} · as of {sig['meta']['asof']:%Y-%m-%d}")
        print(sig["table"].to_string(float_format=lambda v: f"{v:.3f}"))
    elif args.cmd == "retrain":
        from vixster.model import search
        search.run(apply=args.apply)
    elif args.cmd == "stream":
        from vixster.ibkr import stream
        stream.cli()
    elif args.cmd == "gateway":
        from vixster.ibkr import gateway
        print(gateway.status() if args.action == "status" else getattr(gateway, args.action)())


if __name__ == "__main__":
    main()
