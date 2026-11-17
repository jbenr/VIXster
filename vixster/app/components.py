"""Layout pieces in the Viz house style: tables (Viz table_div), alerts, buttons, the shimmer loader."""
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from dash import dcc, html

from vixster.app import theme as T


def table(df: pd.DataFrame, fmt: dict[str, str] | str = "{:,.2f}", cell_style: Callable | None = None,
          index: bool = True, scroll: bool = False, class_name: str = "") -> html.Div:
    """Plain HTML table, Viz table_div look.

    fmt: one format string, or {column: format}. cell_style(row_label, col, value) -> dict
    adds per-cell CSS (long/short shading, red/green returns).
    """
    def _fmt(col, v):
        if hasattr(v, "to_plotly_json"):   # a Dash component (e.g. logo + name) renders as itself
            return v
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return ""
        if isinstance(v, pd.Timestamp):
            return v.strftime("%Y-%m-%d %H:%M:%S") if (v.hour or v.minute or v.second) else v.strftime("%Y-%m-%d")
        if isinstance(v, (bool, np.bool_)):
            return str(v)
        if isinstance(v, (int, float, np.integer, np.floating)):
            f = fmt.get(col) if isinstance(fmt, dict) else fmt
            return f.format(v) if f else str(v)
        return str(v)

    def _label(v):
        return _fmt(None, v) if isinstance(v, pd.Timestamp) else str(v)

    head = ([html.Th(df.index.name or "")] if index else []) + [html.Th(str(c)) for c in df.columns]
    body = []
    for label, row in df.iterrows():
        cells = [html.Th(_label(label), className="row-label")] if index else []
        for c in df.columns:
            st = cell_style(label, c, row[c]) if cell_style else None
            cells.append(html.Td(_fmt(c, row[c]), style=st or None))
        body.append(html.Tr(cells))
    return html.Div(html.Table([html.Thead(html.Tr(head)), html.Tbody(body)], className=f"viz-table {class_name}".strip()),
                    className="table-wrap scroll" if scroll else "table-wrap")


def green_red(v) -> dict | None:
    """The old app's _green_red: green when positive, red otherwise."""
    if not isinstance(v, (int, float, np.integer, np.floating)) or pd.isna(v):
        return None
    return {"color": T.GREEN if v > 0 else T.RED}


def card(*children, title: str | None = None, right=None, class_name: str = "", **kw) -> html.Div:
    """A titled block (uppercase title left, controls right) sitting straight on the page, no box behind it."""
    head = html.Div([html.Div(title, className="card-title"), html.Div(right, className="card-right")],
                    className="card-head") if (title or right) else None
    return html.Div([head, html.Div(children, className="card-body")],
                    className=f"card card-bare {class_name}".strip(), **kw)


def spinner(text: str) -> html.Div:
    """st.spinner: a small turning ring and a message."""
    return html.Div([html.Span(className="spin"), html.Span(text)], className="spinner")


def status(label: str, lines: list[str], state: str = "running") -> html.Details:
    """st.status: a box with a spinner and the steps as they happen. When done it
    collapses to one line with a check (or a red cross); click it to see the steps.
    state: running | complete | error."""
    icon = {"running": html.Span(className="spin"), "complete": html.Span("✓", className="status-icon"),
            "error": html.Span("✕", className="status-icon")}[state]
    return html.Details([
        html.Summary([icon, html.Span(label, className="status-label")], className="status-head"),
        html.Div([html.Div(line, className="status-line") for line in lines], className="status-body"),
    ], open=state == "running", className=f"status status-{state}")


def note(text, bad: bool = False, id: str | None = None) -> html.Div:
    kw = {"id": id} if id else {}
    return html.Div(text, className="note note-bad" if bad else "note", **kw)


def alert(text, kind: str = "error") -> html.Div:
    """A plain line of text (red for errors, amber for warnings), no box."""
    return html.Div(text, className=f"msg msg-{kind}")


def button(text: str, id: str, primary: bool = False) -> html.Button:
    return html.Button(text, id=id, n_clicks=0, className="btn btn-primary" if primary else "btn")


def shimmer(text: str, caption: str | None = None) -> html.Div:
    """Views' shimmer loader: a burgundy/gold sweep across a word (MODELO while the model runs)."""
    return html.Div([html.Div(text, className="shimmer-loader__mark shimmer-loader__mark--text"),
                     html.Span(caption, className="shimmer-loader__caption") if caption else None],
                    className="shimmer-loader")


def loading(child, text: str | None = None, caption: str | None = None) -> dcc.Loading:
    """While a callback runs: the MODELO-style shimmer if `text` is given, otherwise the old content just dims."""
    return dcc.Loading(child, custom_spinner=shimmer(text, caption=caption) if text else html.Span(),
                       delay_show=150, overlay_style={"visibility": "visible", "opacity": 0.35})


RANGES = ["1M", "3M", "6M", "YTD", "1Y", "2Y", "5Y", "10Y", "ALL"]   # Viz RANGE_BUTTON_LABELS


def ranges(id: str, value: str = "ALL", options: list[str] = RANGES) -> dcc.RadioItems:
    """Viz's time-frame buttons: a row of small buttons, the active one gold."""
    return dcc.RadioItems(id=id, options=options, value=value, inline=True, className="viz-ranges",
                          inputClassName="viz-range-input", labelClassName="viz-range-btn")


def range_start(end: pd.Timestamp, label: str, first: pd.Timestamp) -> pd.Timestamp:
    """First date shown for a range button (Viz _range_window)."""
    end = pd.Timestamp(end)
    if label == "YTD":
        start = pd.Timestamp(end.year, 1, 1)
    elif label.endswith("M"):
        start = end - pd.DateOffset(months=int(label[:-1]))
    elif label.endswith("Y"):
        start = end - pd.DateOffset(years=int(label[:-1]))
    else:
        start = first
    return max(start, pd.Timestamp(first))


def ranges_for(first: pd.Timestamp, last: pd.Timestamp) -> list[str]:
    """Only the buttons the history is long enough for (5Y on two years of data would just be ALL)."""
    span = pd.Timestamp(last) - pd.Timestamp(first)
    keep = []
    for r in RANGES:
        if r.endswith("Y") and r != "1Y" and r[:-1].isdigit() and span < pd.Timedelta(days=365 * int(r[:-1]) * 0.9):
            continue
        keep.append(r)
    return keep
