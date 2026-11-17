"""Performance tab: Refresh NAV Data, returns tables, Model vs SPX charts, the lads."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
from dash import Input, Output, dcc, html, no_update

from vixster import config
from vixster.app import charts
from vixster.app import components as C
from vixster.app import jobs
from vixster.app import theme as T
from vixster.app.cache import by_files
from vixster.io import read_parquet
from vixster.perf import gitpush, nav

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
FILES = (config.NAV, config.FRED)


def layout():
    return html.Div(className="stack", children=[
        html.Div([html.H2("Historical Performance"),
                  html.Div([html.Span(id="perf-asof", className="asof"), C.button("Refresh NAV Data", "perf-refresh")],
                           className="section-right")], className="section-head"),
        html.Div(id="perf-status"),
        html.Div(id="perf-body"),
        html.Div([html.H2("The Lads")], className="section-head"),
        html.Div(id="perf-lads"),
        html.Div([html.H2("Model vs SPX"),
                  C.ranges("perf-range", "ALL")],
                 className="section-head"),
        C.loading(html.Div(id="perf-charts")),
        dcc.Interval(id="perf-poll", interval=400, disabled=True),
    ])


@by_files(*FILES, maxsize=4)
def _load():
    df, investors = nav.load_performance_from_disk()
    return df, investors


def _monthly(df: pd.DataFrame) -> pd.Series:
    return nav.period_returns(df, "ME") * 100


def _returns(df: pd.DataFrame):
    m = _monthly(df)
    grid = (pd.DataFrame({"Year": m.index.year, "Month": m.index.strftime("%b"), "r": m.values})
            .pivot_table(index="Year", columns="Month", values="r", aggfunc="first").reindex(columns=MONTHS))
    yearly = (nav.period_returns(df, "YE") * 100).rename("Return").to_frame()
    yearly.index = yearly.index.year
    yearly["Sharpe"] = nav.yearly_sharpe(df)
    yearly = yearly.T
    yearly.columns = yearly.columns.astype(str)
    total = pd.DataFrame({"Total Return": [df["Cumulative Return"].iloc[-1] * 100],
                          "Sharpe": [nav.sharpe(df["Daily Return"])]}, index=["Model"])
    gr = lambda _r, _c, v: C.green_red(v)  # noqa: E731

    def yearly_table():
        # Return row as %, Sharpe row as a ratio
        t = yearly.copy().astype(object)
        for c in t.columns:
            r, s = yearly.loc["Return", c], yearly.loc["Sharpe", c]
            t.loc["Return", c] = "" if pd.isna(r) else f"{r:.1f}%"
            t.loc["Sharpe", c] = "" if pd.isna(s) else f"{s:.2f}"
        return C.table(t, cell_style=lambda r, c, _v: C.green_red(yearly.loc[r, c]))

    grid.index = grid.index.astype(str)
    grid.index.name = ""
    return C.card(
        C.table(grid, fmt="{:.1f}%", cell_style=gr, class_name="returns-grid"),
        html.Div([yearly_table(),
                  C.table(total, fmt={"Total Return": "{:.1f}%", "Sharpe": "{:.2f}"}, cell_style=gr)],
                 className="tables-row"),
        title="Returns", class_name="card-bare card-fit")


@by_files(*FILES, maxsize=32)
def _charts(rng: str):
    df, _ = _load()
    feat = read_parquet(config.FRED)
    feat.index = pd.to_datetime(feat.index)
    dates = pd.to_datetime(df["Date"])
    start = C.range_start(dates.max(), rng, dates.min())
    monthly = _monthly(df)
    m_f = monthly[monthly.index >= start]

    # cumulative, both lines starting from the first month shown
    model_cum = (1 + m_f / 100).cumprod() - 1
    spx = feat["SP500"].resample("ME").last()
    spx = spx.loc[model_cum.index.intersection(spx.index)]
    spx_cum = (1 + spx.pct_change().fillna(0)).cumprod() - 1
    cum = pd.DataFrame({"Model": model_cum, "SP500": spx_cum}) * 100
    cum.index = cum.index.strftime("%b %y")

    # monthly model vs SPX (SPX month-end to month-end)
    spx_m = feat["SP500"].dropna().resample("ME").last().pct_change() * 100
    rel = pd.DataFrame({"Model": m_f, "SPX": spx_m}).dropna()
    rel = rel.loc[rel.index.intersection(model_cum.index.intersection(spx.index))]
    rel.index = rel.index.strftime("%b %y")

    # drawdown from the daily NAV returns, peak-to-date within the period shown
    daily = df.set_index("Date")["Daily Return"]
    daily.index = pd.to_datetime(daily.index)
    daily = daily[daily.index >= start]
    wealth = (1 + daily.fillna(0)).cumprod()
    dd = ((wealth / wealth.cummax() - 1) * 100).rename("Drawdown").to_frame()

    return html.Div([
        html.Div("Cumulative Monthly Returns: Model vs SPX", className="sub-title"),
        charts.cumulative(cum, "Cumulative Return", [T.MODEL_C, T.SPX_C]),
        html.Div("Monthly Returns: Model vs SPX  ·  Relative Performance (Model − SPX)", className="sub-title"),
        charts.monthly_vs(rel, [T.MODEL_C, T.SPX_C]),
        html.Div(f"Drawdown · max {dd['Drawdown'].min():.1f}%", className="sub-title"),
        charts.timeseries(dd, [T.RED], height=2.0, fmt="{:.1f}%", fill=True, yfmt="{:.0f}%", ylabel="Drawdown"),
    ])


def _lads(investors: dict):
    if not investors:
        return C.note('Investor details not yet populated. Click "Refresh NAV Data" to fetch investor info.')
    rows = []
    for name, info in investors.items():
        logo = html.Img(src=info["logo_url"], className="lad-logo", alt="") if info.get("logo_url") else None
        rows.append({"Name": html.Span([logo, name], className="lad-cell"),
                     "Initial Investment": info["initial_investment"], "Ownership": info["ownership_pct"] * 100,
                     "Current Value": info["current_value"], "PnL": info["pnl"], "Returns %": info["return_pct"]})
    fmt = {"Initial Investment": "${:,.2f}", "Ownership": "{:.2f}%", "Current Value": "${:,.2f}",
           "PnL": "${:,.2f}", "Returns %": "{:.1f}%"}
    return C.table(pd.DataFrame(rows), fmt=fmt, index=False, class_name="lads-table",
                   cell_style=lambda _r, c, v: C.green_red(v) if c in ("PnL", "Returns %") else None)


def _refresh(log):
    """What the live app's Refresh NAV Data does: pull NAV, then push to GitHub."""
    df, investors = nav.pull_performance(log=log)
    log("Pushing to GitHub…")
    gitpush.push(f"auto push on NAV ref {datetime.now()}")
    return df["Date"].max()


def register(app):
    @app.callback(Output("perf-poll", "disabled"), Output("perf-status", "children", allow_duplicate=True),
                  Input("perf-refresh", "n_clicks"), prevent_initial_call=True)
    def _start(_):
        jobs.start("nav", _refresh)
        return False, C.status("Refreshing NAV data…", jobs.messages(jobs.get("nav")))

    @app.callback(Output("perf-status", "children"), Output("perf-poll", "disabled", allow_duplicate=True),
                  Output("perf-body", "children"), Output("perf-lads", "children"),
                  Output("perf-range", "options"), Output("perf-asof", "children"),
                  Input("perf-poll", "n_intervals"), prevent_initial_call="initial_duplicate")
    def _poll(n):
        job = jobs.get("nav")
        if job["running"]:
            return C.status("Refreshing NAV data…", jobs.messages(job)), False, \
                no_update, no_update, no_update, no_update
        status = ""
        if n and job["finished"]:   # the box only shows after a click, like the live app
            if job["ok"]:
                done = job["result"]
                if isinstance(done, tuple):   # a scheduled run returns (df, investors)
                    done = done[0]["Date"].max()
                label = f"NAV data updated through {pd.Timestamp(done):%b %d, %Y}."
                status = C.status(label, jobs.messages(job), "complete")
            else:
                status = C.status(f"Error updating performance data: {job['error']}", jobs.messages(job), "error")
        try:
            df, investors = _load()
        except Exception as e:
            return status or C.alert(f"Could not load performance data: {e}", "warning"), True, "", "", C.RANGES, ""
        if df.empty:
            return C.alert('Loaded performance data is empty. Try clicking "Refresh NAV Data" again.', "error"), \
                True, "", "", C.RANGES, ""
        gaps = nav.find_gaps(df)
        gap = C.alert("NAV history has gaps (returns may be wrong) between: " +
                      ", ".join(f"{a:%Y-%m-%d} → {b:%Y-%m-%d}" for a, b in gaps) +
                      ". Click 'Refresh NAV Data' to backfill.", "error") if gaps else None
        dates = pd.to_datetime(df["Date"])
        asof = f"NAV as of {pd.Timestamp(df['Date'].max()):%b %d, %Y}"
        return (status, True, html.Div([gap, _returns(df)], className="stack"), _lads(investors),
                C.ranges_for(dates.min(), dates.max()), asof)

    @app.callback(Output("perf-charts", "children"), Input("perf-range", "value"), Input("perf-body", "children"))
    def _chart(rng, _):
        try:
            return _charts(rng or "ALL")
        except Exception as e:
            return C.alert(f"Could not draw the charts: {e}", "warning")
