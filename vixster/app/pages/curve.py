"""Curve PCA tab: principal components of the VX futures curve (F1..F8).

What the curve is doing (level / slope / curvature), how much each factor
explains, and where today's curve sits against its PCA fair value, per contract
and per calendar spread.
"""
from __future__ import annotations

import pandas as pd
from dash import Input, Output, dcc, html

from vixster import config
from vixster.app import charts
from vixster.app import components as C
from vixster.app import theme as T
from vixster.app.cache import by_files
from vixster.model import pca

PC_COLORS = [T.ORANGE, T.GOLD, "#5DADE2", "#A569BD"]


def _seg(id: str, options: list, value) -> dcc.RadioItems:
    return dcc.RadioItems(id=id, options=options, value=value, inline=True, className="seg",
                          inputClassName="seg-input", labelClassName="seg-btn")


def layout():
    controls = html.Div([
        html.Div([html.Span("Window", className="ctl-label"), C.ranges("pca-window", "2Y", list(pca.WINDOWS))]),
        html.Div([html.Span("Fit on", className="ctl-label"),
                  _seg("pca-basis", [{"label": "Levels", "value": "levels"},
                                     {"label": "Daily changes", "value": "changes"}], "levels")]),
        html.Div([html.Span("Factors", className="ctl-label"), _seg("pca-k", [2, 3, 4], 3)]),
    ], className="controls")
    return html.Div(className="stack", children=[
        html.Div([html.H2("Curve PCA"), controls], className="section-head section-head--wrap"),
        C.loading(html.Div(id="pca-body"), "PCA"),
    ])


def _stat(label, value, sub=None, tone=None):
    return html.Div([html.Div(label, className="stat-label"), html.Div(value, className="stat-value", style=tone),
                     html.Div(sub, className="stat-sub") if sub else None], className="stat")


def _z_style(_row, col, v):
    if col != "Z" or not isinstance(v, float) or pd.isna(v):
        return C.green_red(v) if col == "Resid" else None
    return {**(C.green_red(v) or {}), "fontWeight": "bold"} if abs(v) >= 1.5 else C.green_red(v)


@by_files(config.VX_CURVE, maxsize=32)
def _render(window: str, basis: str, k: int):
    r = pca.fit(pca.load_curve(), window, k, changes=basis == "changes")
    ex = r["explained"]
    today, spreads = r["today"], r["spreads"]
    sign = lambda s: [T.GREEN if v > 0 else T.RED for v in s]  # noqa: E731

    stats = html.Div([
        *[_stat(f"PC{i + 1} · {pca.NAMES[i]}", f"{ex.iloc[i] * 100:.1f}%", "of variance") for i in range(min(k, 3))],
        _stat(f"First {k} together", f"{ex.iloc[:k].sum() * 100:.1f}%", "of variance", {"color": T.GOLD}),
        _stat("Sample", f"{r['n_obs']:,} days", f"{r['start']:%b %Y} – {r['asof']:%b %d, %Y}"),
    ], className="stats")

    return html.Div(className="stack", children=[
        stats,
        html.Div([
            C.card(charts.category_lines(r["loadings"], "Loading", PC_COLORS, xlabel="Contract"),
                   title="Factor loadings"),
            C.card(charts.bars(ex * 100, "Variance (%)", [T.ORANGE] * k + [T.GREY] * (len(ex) - k), fmt="{:.1f}%",
                               xlabel="Component"), title="Variance explained"),
        ], className="grid-2"),
        html.Div([
            C.card(charts.category_lines(today[["Actual", "PCA fair"]], "VX", [T.ACTUAL_C, T.PRED_C],
                                         xlabel="Contract"),
                   title=f"Curve vs PCA fair value · {r['asof']:%Y-%m-%d}"),
            C.card(charts.bars(today["Resid"], "Actual − fair", sign(today["Resid"]), xlabel="Contract"),
                   title="Residual per contract"),
        ], className="grid-2"),
        html.Div([
            C.card(C.table(spreads, fmt={"Actual": "{:+.2f}", "PCA fair": "{:+.2f}", "Resid": "{:+.3f}",
                                         "Z": "{:+.2f}"}, cell_style=_z_style),
                   html.Div("Resid = actual spread − PCA fair spread. Positive: the spread is rich to the curve's "
                            "factors; negative: cheap. Z against the spread's own residual history in the window.",
                            className="note"),
                   title="Calendar spreads vs PCA", class_name="card-bare"),
            C.card(charts.bars(spreads["Z"], "Residual z", sign(spreads["Z"])), title="Spread residual z-scores"),
        ], className="grid-2"),
        C.card(*[html.Div([html.Div(c, className="sub-title"),
                           charts.timeseries(r["scores"][[c]], [PC_COLORS[i]])])
                 for i, c in enumerate(r["scores"].columns)],
               title="Factor scores over time"),
    ])


def register(app):
    @app.callback(Output("pca-body", "children"), Input("pca-window", "value"), Input("pca-basis", "value"),
                  Input("pca-k", "value"))
    def _pca(window, basis, k):
        try:
            return _render(window or "2Y", basis or "levels", int(k or 3))
        except Exception as e:
            return C.alert(f"Could not run the PCA: {e}", "error")
