"""Model tab: Run Modelo — today's signal table, residuals, z-scores, actual vs predicted, history,
and the strategy's backtest stats underneath (click a section to open it)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from dash import Input, Output, html

from vixster import config
from vixster.app import charts
from vixster.app import components as C
from vixster.app import theme as T
from vixster.app.cache import by_files
from vixster.model import service, strategy

DATA_FILES = (config.VX_SPREADS, config.VX_CURVE, config.FRED, config.MODELS)


def layout():
    return html.Div(className="stack", children=[
        html.Div([
            html.Div([html.H2("Modelo"), html.Img(src="/assets/modelo.png", className="modelo-logo", alt="Modelo")],
                     className="modelo-title"),
            C.button("Run Modelo", "sig-run", primary=True),
        ], className="section-head"),
        C.loading(html.Div(id="sig-body"), "MODELO"),
    ])


def _side_style(signal: pd.Series):
    def style(_row, col, _v):
        side = signal.get(col, "")
        return ({"background": T.LONG_BG} if side == "Long" else
                {"background": T.SHORT_BG} if side == "Short" else None)
    return style


def _trade(signal: pd.Series):
    long_ = [s for s, v in signal.items() if v == "Long"]
    short = [s for s, v in signal.items() if v == "Short"]
    if not long_ and not short:
        return html.Span("NO TRADE", className="chip")
    return html.Span([html.Span(f"LONG {long_[0]}", className="chip chip-long") if long_ else None,
                      html.Span(f"SHORT {short[0]}", className="chip chip-short") if short else None])


def _body(sig: dict):
    res: pd.DataFrame = sig["result"]
    tab: pd.DataFrame = sig["table"]
    date = pd.Timestamp(res.index[-1])
    signal = tab.loc["Signal"]
    colors = [T.GREEN if signal.get(s) == "Long" else T.RED if signal.get(s) == "Short" else T.GREY
              for s in tab.columns]
    avp = pd.DataFrame({"Actual": tab.loc["Actual"].astype(float), "Predicted": tab.loc["Pred"].astype(float)})
    return html.Div(className="stack", children=[
        C.card(C.table(tab, fmt="{:.2f}", cell_style=_side_style(signal), class_name="signal-table"),
               title=f"Signal {date:%Y-%m-%d}", right=_trade(signal), class_name="card-fit"),
        html.Div([
            C.card(charts.bars(tab.loc["Resid"].astype(float), "Net Residual", colors), title="Net Residual per Spread"),
            C.card(charts.bars(tab.loc["Z"].astype(float), "Z-Score", colors), title="Z-Scores per Spread"),
        ], className="grid-2"),
        C.card(charts.category_lines(avp, "Spread Value", [T.ACTUAL_C, T.PRED_C], width=13, height=2.4),
               title="Actual Spreads vs Predicted Spreads"),
        C.card(C.table(res.tail(50).iloc[::-1], fmt="{:.4f}", scroll=True), title="Historical Signals"),
        C.card(C.table(sig["data"].tail(5).iloc[::-1], fmt="{:.4f}", scroll=True), title="Latest Data"),
        _stats(res),
    ])


# ── strategy stats ───────────────────────────────────────────────────────────

def _section(title: str, *children, open: bool = False) -> html.Details:
    return html.Details([html.Summary(title, className="fold-title"), html.Div(children, className="fold-body")],
                        open=open, className="fold")


def _pnl(_r, col, v):
    return C.green_red(v) if col in ("P&L", "Total P&L", "PnL", "Long P&L", "Short P&L", "Avg Trade") else None


def _summary(res: pd.DataFrame):
    m = strategy.metrics(res)
    perf = pd.DataFrame([{"Total P&L": m["TotalPnL"], "Annual P&L": m["AnnualReturn"], "Volatility": m["Volatility"],
                          "Sharpe": m["Sharpe"], "Sortino": m["Sortino"], "Max Drawdown": m["MaxDrawdown"],
                          "Calmar": m["Calmar"]}], index=["Strategy"])
    trade = pd.DataFrame([{"Trades": m["Trades"], "Win Rate": m["WinRate"] * 100, "Avg Win": m["AvgWin"],
                           "Avg Loss": m["AvgLoss"], "Avg Days Held": m["AvgDays"],
                           "Time in Market": m["TimeInMarket"] * 100}], index=["Strategy"])
    first, last = res.index[0], res.index[-1]
    return html.Div([
        C.table(perf, fmt={"Total P&L": "{:+.2f}", "Annual P&L": "{:+.2f}", "Volatility": "{:.2f}",
                           "Sharpe": "{:.2f}", "Sortino": "{:.2f}", "Max Drawdown": "{:.2f}", "Calmar": "{:.2f}"},
                cell_style=lambda _r, c, v: C.green_red(v) if c in ("Total P&L", "Annual P&L", "Max Drawdown") else None),
        C.table(trade, fmt={"Trades": "{:,.0f}", "Win Rate": "{:.1f}%", "Avg Win": "{:+.3f}", "Avg Loss": "{:+.3f}",
                            "Avg Days Held": "{:.1f}", "Time in Market": "{:.0f}%"},
                cell_style=lambda _r, c, v: C.green_red(v) if c in ("Avg Win", "Avg Loss") else None),
        C.note(f"Backtest {first:%b %Y} – {last:%b %d, %Y}. P&L in spread points, one lot per leg."),
    ], className="tables-col")


def _by_year(res: pd.DataFrame):
    tr = strategy.trades(res)
    rows = {}
    for y, g in res.groupby(res.index.year):
        r = g["DailyPnL"]
        cum = r.cumsum()
        t = tr[pd.to_datetime(tr["Exit"]).dt.year == y]
        rows[str(y)] = {"P&L": r.sum(), "Sharpe": r.mean() / r.std() * np.sqrt(252) if r.std() else np.nan,
                        "Max Drawdown": (cum - cum.cummax().clip(lower=0)).min(), "Trades": len(t),
                        "Win Rate": (t["PnL"] > 0).mean() * 100 if len(t) else np.nan,
                        "Avg Trade": t["PnL"].mean() if len(t) else np.nan}
    df = pd.DataFrame(rows).T.iloc[::-1]
    df.index.name = "Year"
    return C.table(df, fmt={"P&L": "{:+.2f}", "Sharpe": "{:.2f}", "Max Drawdown": "{:.2f}", "Trades": "{:,.0f}",
                            "Win Rate": "{:.1f}%", "Avg Trade": "{:+.3f}"}, cell_style=_pnl)


def _by_spread(res: pd.DataFrame):
    tr = strategy.trades(res)
    on = res[res["Trade"]]
    rows = {}
    for s in config.TRADED:
        lp = on.loc[on["LongSpread"] == s, "PnL_Long"].sum()
        sp = on.loc[on["ShortSpread"] == s, "PnL_Short"].sum()
        t = tr[(tr["Long"] == s) | (tr["Short"] == s)]
        rows[s] = {"Times Long": (tr["Long"] == s).sum(), "Times Short": (tr["Short"] == s).sum(),
                   "Long P&L": lp, "Short P&L": sp, "Total P&L": lp + sp,
                   "Win Rate": (t["PnL"] > 0).mean() * 100 if len(t) else np.nan}
    df = pd.DataFrame(rows).T
    df.index.name = "Spread"
    return C.table(df, fmt={"Times Long": "{:,.0f}", "Times Short": "{:,.0f}", "Long P&L": "{:+.2f}",
                            "Short P&L": "{:+.2f}", "Total P&L": "{:+.2f}", "Win Rate": "{:.1f}%"}, cell_style=_pnl)


def _recent_trades(res: pd.DataFrame, n: int = 30):
    tr = strategy.trades(res).tail(n).iloc[::-1].set_index("TradeID")
    tr["Entry"] = pd.to_datetime(tr["Entry"]).dt.strftime("%Y-%m-%d")
    tr["Exit"] = pd.to_datetime(tr["Exit"]).dt.strftime("%Y-%m-%d")
    tr.index.name = "Trade"
    return C.table(tr, fmt={"Days": "{:.0f}", "PnL": "{:+.3f}"}, cell_style=_pnl, scroll=True)


@by_files(*DATA_FILES, maxsize=16)
def _equity(rng: str):
    res = service.history()["result"]
    idx = pd.to_datetime(res.index)
    start = C.range_start(idx.max(), rng, idx.min())
    r = res.loc[idx >= start, "DailyPnL"].fillna(0)
    cum = r.cumsum()
    dd = cum - cum.cummax().clip(lower=0)
    return html.Div([
        html.Div(f"Cumulative P&L · {cum.iloc[-1]:+.2f}", className="sub-title"),
        charts.timeseries(cum.rename("P&L").to_frame(), [T.GOLD], height=2.2, fmt="{:+.2f}", ylabel="Points"),
        html.Div(f"Drawdown · max {dd.min():.2f}", className="sub-title"),
        charts.timeseries(dd.rename("Drawdown").to_frame(), [T.RED], height=1.6, fmt="{:.2f}", fill=True,
                          ylabel="Points"),
    ])


def _stats(res: pd.DataFrame):
    idx = pd.to_datetime(res.index)
    return html.Div(className="stack", children=[
        html.H2("Strategy Stats", className="stats-head"),
        _section("Summary", _summary(res), open=True),
        _section("Equity curve & drawdown",
                 html.Div(C.ranges("bt-range", "ALL", C.ranges_for(idx.min(), idx.max())), className="fold-controls"),
                 C.loading(html.Div(id="bt-equity"))),
        _section("By year", _by_year(res)),
        _section("By spread", _by_spread(res)),
        _section("Recent trades", _recent_trades(res)),
    ])


def register(app):
    @app.callback(Output("sig-body", "children"), Input("sig-run", "n_clicks"), prevent_initial_call=True)
    def _run(_):
        try:
            return _body(service.live_signal(use_live=True))
        except Exception as e:
            return C.alert(f"Error computing live signal: {e}")

    @app.callback(Output("bt-equity", "children"), Input("bt-range", "value"))
    def _eq(rng):
        try:
            return _equity(rng or "ALL")
        except Exception as e:
            return C.alert(f"Could not draw the equity curve: {e}")
