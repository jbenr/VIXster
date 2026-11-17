"""Production entry point: the app, the daily auto-refresh, and a cache warm-up."""
import threading

from vixster.app.app import server  # noqa: F401
from vixster.app import scheduler


def _warm():
    """Compute the model history and the default charts once, so the first visit is quick."""
    try:
        from vixster.app.pages import performance
        from vixster.model import service
        service.history()
        performance._charts("ALL")
    except Exception:
        pass


scheduler.start()
threading.Thread(target=_warm, daemon=True, name="warm").start()
