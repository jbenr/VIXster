"""Commit and push the VIXster repo after a NAV refresh, like the live app's
utils.git_push: `git add .`, commit, push to origin main. Output goes to the
server log; a failed step (e.g. nothing to commit) doesn't stop the refresh."""
from __future__ import annotations

import subprocess

from vixster import config


def push(message: str) -> None:
    for cmd in (["git", "add", "."], ["git", "commit", "-m", message], ["git", "push", "origin", "main"]):
        r = subprocess.run(cmd, cwd=config.ROOT, capture_output=True, text=True)
        print(r.stdout if r.returncode == 0 else f"❗️ Error: {r.stderr}", flush=True)
