"""Data tab: live spread quotes, IBKR login, historical spreads + FRED features, Update Historical Data."""
from __future__ import annotations

import pandas as pd
from dash import Input, Output, dcc, html, no_update

from vixster import config
from vixster.app import components as C
from vixster.app import jobs
from vixster.data import live, pipeline
from vixster.ibkr import gateway, procs
from vixster.io import read_parquet

MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()


def layout():
    return html.Div(className="stack", children=[
        C.card(html.Div(id="gw-msg"), html.Div(id="spread-table"),
               title="Live Spread Quotes", class_name="card-fit",
               right=[C.button("Refresh Spread File", "spread-refresh"), C.button("IBKR Login", "gw-login")]),
        html.H2("Historical Data"),
        html.Div([
            C.card(html.Div(id="hist-spreads"), title="Historical VIX Spreads", right=html.Span(id="hist-spreads-asof")),
            C.card(html.Div(id="hist-fred"), title="Historical Feature Data", right=html.Span(id="hist-fred-asof")),
        ], className="row-fit"),
        # Like the live (Streamlit) app: the button under the tables, then a status box
        # that lists each step as it runs and collapses to a summary when done.
        html.Div(C.button("Update Historical Data", "data-update"), className="update-row"),
        html.Div(id="data-status"),
        dcc.Interval(id="data-poll", interval=400, disabled=True),
        dcc.Store(id="data-done"),
    ])


def _data_label(job: dict) -> tuple[str, str]:
    """Summary line and state for a finished update, worded like the live app."""
    r = job["result"] or {}
    if not job["ok"]:
        return f"Data update failed: {job['error']}", "error"
    if r.get("errors"):
        return "Data update failed: " + "; ".join(r["errors"]), "error"
    label = f"Historical data updated through {r['through']:%b %d, %Y} ({r['seconds']:.1f} s)."
    if r.get("failed"):
        label += f" {len(r['failed'])} contract(s) failed to download."
    return label, "complete"


def _months(expiry: str) -> str:
    """'20250416/20250521' -> 'Apr25/May25'."""
    def one(code):
        code = str(code)
        return f"{MONTHS[int(code[4:6]) - 1]}{code[2:4]}" if code[:6].isdigit() else code
    return "/".join(one(c) for c in str(expiry).split("/"))


def _spread_table():
    """The streamer's file, shaped like the old spreads.parquet."""
    q = live.quotes()
    sp = q[q["Kind"] == "spread"] if q is not None else None
    if sp is None or sp.empty:
        return C.note("No spread data loaded yet.", )
    df = pd.DataFrame({
        "Contracts": sp["Contract"].to_numpy(),
        "Months": [_months(e) for e in sp["Expiry"]],
        "Bid Size": sp["Bid Size"].to_numpy(),
        "Bid Price": sp["Bid"].to_numpy(),
        "Ask Price": sp["Ask"].to_numpy(),
        "Ask Size": sp["Ask Size"].to_numpy(),
        "Last Update": pd.to_datetime(sp["Updated"]).dt.strftime("%Y-%m-%d %H:%M:%S").to_numpy(),
    }, index=pd.Index(sp["Key"].to_numpy(), name="Spread"))
    return html.Div([
        C.table(df, fmt={"Bid Price": "{:.2f}", "Ask Price": "{:.2f}", "Bid Size": "{:.0f}", "Ask Size": "{:.0f}"}),
        C.note(f"Last Update: {df['Last Update'].max()}"),
    ])


def _history(path, cols=None, fmt="{:.2f}"):
    try:
        df = read_parquet(path)
    except FileNotFoundError:
        return C.note(f"No file found at {path}."), ""
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    last = df.index.max()
    stale = pd.Timestamp.now() - last >= pd.Timedelta("2 days")
    show = (df[cols] if cols else df).tail(10)
    show.index.name = "Date"
    return C.table(show, fmt=fmt), html.Span(f"Last Update: {last:%Y-%m-%d}",
                                             className="asof asof-bad" if stale else "asof")


def register(app):
    @app.callback(Output("spread-table", "children"), Input("spread-refresh", "n_clicks"))
    def _spreads(_):
        return _spread_table()

    @app.callback(Output("gw-msg", "children"), Input("gw-login", "n_clicks"), prevent_initial_call=True)
    def _login(_):
        try:
            msgs = [gateway.start(), procs.start_streamer()]   # the streamer waits for the gateway by itself
        except Exception as e:
            return C.alert(f"Failed to launch IBKR Gateway. {e}")
        return C.note(" ".join(m for m in msgs if m))

    @app.callback(Output("hist-spreads", "children"), Output("hist-spreads-asof", "children"),
                  Output("hist-fred", "children"), Output("hist-fred-asof", "children"),
                  Input("data-done", "data"))
    def _hist(_):
        return *_history(config.VX_SPREADS, config.SPREADS), *_history(config.FRED, fmt="{:,.2f}")

    @app.callback(Output("data-status", "children"), Output("data-poll", "disabled"),
                  Input("data-update", "n_clicks"), prevent_initial_call=True)
    def _update(_):
        """Start the update (through the job runner, so it can't overlap a scheduled run)."""
        jobs.start("data", lambda log: pipeline.run(log=log))
        return C.status("Updating historical data…", jobs.messages(jobs.get("data"))), False

    @app.callback(Output("data-status", "children", allow_duplicate=True),
                  Output("data-poll", "disabled", allow_duplicate=True), Output("data-done", "data"),
                  Input("data-poll", "n_intervals"), prevent_initial_call=True)
    def _poll(_):
        job = jobs.get("data")
        if job["running"]:
            return C.status("Updating historical data…", jobs.messages(job)), False, no_update
        label, state = _data_label(job)
        return C.status(label, jobs.messages(job), state), True, job["finished"]
