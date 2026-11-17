"""Parquet helpers: atomic writes and a small mtime-keyed read cache.

Atomic writes matter because the app reads files that the streamer and the
data pull rewrite. Writing to a temp file and os.replace()-ing it means a
reader sees the old file or the new one, never a half-written one.
"""
from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path

import pandas as pd

_CACHE: dict[str, tuple[float, pd.DataFrame]] = {}
_LOCK = threading.Lock()


def write_parquet(df: pd.DataFrame, path: Path, **kwargs) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    os.close(fd)
    try:
        df.to_parquet(tmp, **kwargs)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_parquet(path: Path) -> pd.DataFrame:
    """Read a parquet, re-using the last read if the file hasn't changed.

    Raises FileNotFoundError for a missing or empty file (the "not ready yet"
    case) so callers can tell that apart from a corrupt file.
    """
    path = Path(path)
    try:
        st = path.stat()
    except FileNotFoundError:
        raise FileNotFoundError(f"{path.name} does not exist yet") from None
    if st.st_size == 0:
        raise FileNotFoundError(f"{path.name} is empty")
    key = str(path)
    with _LOCK:
        hit = _CACHE.get(key)
        if hit and hit[0] == st.st_mtime:
            return hit[1].copy()
    df = pd.read_parquet(path)
    with _LOCK:
        _CACHE[key] = (st.st_mtime, df)
    return df.copy()
