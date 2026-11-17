"""Tiny memo for page renders: results are reused until the data files change."""
from __future__ import annotations

import functools
import threading
from pathlib import Path


def stamp(*paths: Path) -> tuple:
    return tuple(p.stat().st_mtime if p.exists() else 0 for p in paths)


def by_files(*paths: Path, maxsize: int = 64):
    """Cache f(*args) until any of `paths` is rewritten."""
    def deco(fn):
        cached = functools.lru_cache(maxsize=maxsize)(lambda st, *a: fn(*a))
        lock = threading.Lock()

        @functools.wraps(fn)
        def wrapper(*args):
            st = stamp(*paths)
            with lock:
                return cached(st, *args)
        wrapper.cache_clear = cached.cache_clear
        return wrapper
    return deco
