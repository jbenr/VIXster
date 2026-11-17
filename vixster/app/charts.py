"""Charts in the Viz (PrismFP) style from Views/utils/viz.py, on a dark canvas.

Same rules as Viz: value axis on the right, bold bottom/right spines with ticks
pointing out, legend under the chart with square swatches, endpoint value flags
in the series colour, green/red bars by sign. Titles live in the card header on
the page rather than inside the image. Drawn with matplotlib, sent as SVG with a
transparent background so the chart sits in its card.
"""
from __future__ import annotations

import base64
import io

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from dash import html  # noqa: E402
from matplotlib.backends.backend_agg import FigureCanvasAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.offsetbox import AnnotationBbox, DrawingArea  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402
from matplotlib.text import Text  # noqa: E402
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MaxNLocator, NullLocator  # noqa: E402

from vixster.app import theme as T  # noqa: E402

matplotlib.rcParams.update({"font.family": "sans-serif",
                            "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
                            "font.size": 8, "svg.fonttype": "none"})


class _Square:
    """Viz legend swatch: a coloured square instead of a line sample."""

    def legend_artist(self, legend, h, fontsize, box):
        color = h.get_color() if hasattr(h, "get_color") else h.get_facecolor()
        a = Rectangle((box.xdescent, box.ydescent), box.height, box.height, facecolor=color, edgecolor="none",
                      transform=box.get_transform())
        box.add_artist(a)
        return a


def _new(width: float, height: float):
    fig = Figure(figsize=(width, height))
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_facecolor(T.PLOT)
    ax.set_axisbelow(True)
    ax.grid(True, color=T.GRID, linewidth=0.7)
    return fig, ax


def _style(ax, ylabel: str | None = None, yfmt: str | None = None, xlabel: str | None = None):
    """Viz._style_ax, dark."""
    if ylabel:
        ax.set_ylabel(ylabel.upper(), fontsize=7, color=T.DIM, labelpad=8)
    if xlabel:
        ax.set_xlabel(xlabel.upper(), fontsize=7, color=T.DIM, labelpad=6)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="both", which="major", labelsize=7.5, colors=T.INK, direction="out", length=3.5, width=1.0)
    ax.tick_params(axis="both", which="minor", colors=T.INK, direction="out", length=2.5, width=0.6)
    for s in ("top", "left"):
        ax.spines[s].set_visible(False)
    for s in ("bottom", "right"):
        ax.spines[s].set_color(T.INK)
        ax.spines[s].set_linewidth(1.1)
    if yfmt:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: yfmt.format(v)))


def _legend(ax, y: float | None = None):
    """Viz puts the legend under the chart; here it sits above the plot, left, so it never collides with ticks."""
    if ax.get_legend_handles_labels()[1]:
        ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, borderaxespad=0.3, fontsize=7.5, frameon=False, labelcolor=T.TEXT,
                  handlelength=0.9, handleheight=0.9, handler_map={Line2D: _Square()})


def _flag(fig, ax, x, y, text: str, color: str):
    """Viz's left-pointing endpoint flag."""
    props = FontProperties(size=7, weight="bold")
    w, h, _ = fig.canvas.get_renderer().get_text_width_height_descent(text, props, ismath=False)
    w, h = w * 72 / fig.dpi, h * 72 / fig.dpi
    rw, rh, pw = w + 8, h + 5, 4
    da = DrawingArea(pw + rw, rh, 0, 0)
    da.add_artist(Polygon([(0, rh / 2), (pw, 0), (pw + rw, 0), (pw + rw, rh), (pw, rh)], closed=True,
                          facecolor=color, edgecolor=color, linewidth=0.8))
    da.add_artist(Text(pw + rw / 2, rh / 2, text, color=T.BG, fontproperties=props, ha="center", va="center"))
    ax.add_artist(AnnotationBbox(da, (x, y), xybox=(0, 0), xycoords="data", boxcoords="offset points",
                                 box_alignment=(0, 0.5), frameon=False, pad=0, annotation_clip=False, zorder=20))


PHONE_WIDTH = 4.4   # inches; a phone shows the chart at about this size, so text stays readable


def _svg(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", transparent=True, pad_inches=0.08)
    return "data:image/svg+xml;base64," + base64.b64encode(buf.getvalue()).decode()


def _img(fig, alt: str):
    """The chart as drawn, plus a narrower re-layout of wide charts for phones (CSS picks one)."""
    w, h = fig.get_size_inches()
    desk = _svg(fig)
    if w <= 7.5:
        return html.Img(src=desk, alt=alt, className="viz-chart")
    for ax in fig.axes:   # fewer month labels when narrow
        ticks, labels = ax.get_xticks(), [t.get_text() for t in ax.get_xticklabels()]
        if len(ticks) > 6 and any(labels):
            step = int(np.ceil(len(ticks) / 5))
            ax.set_xticks(ticks[::step], labels[::step])
    fig.set_size_inches(PHONE_WIDTH, h * 0.85)
    return html.Div([html.Img(src=desk, alt=alt, className="viz-chart only-desk"),
                     html.Img(src=_svg(fig), alt=alt, className="viz-chart only-phone")])


def _cat_ticks(ax, x, labels, max_labels: int = 14):
    step = max(1, int(np.ceil(len(x) / max_labels)))
    ax.set_xticks(x[::step], [str(l) for l in list(labels)[::step]])
    ax.set_xlim(-0.6, len(x) - 0.4)


# ── chart types ──────────────────────────────────────────────────────────────

def bars(values: pd.Series, ylabel: str, colors: list[str], fmt: str = "{:+.2f}", width=6.4, height=2.5,
         xlabel: str = "Spread"):
    """One bar per spread (net residual, z-score), value printed on each bar."""
    fig, ax = _new(width, height)
    v = values.to_numpy(float)
    x = np.arange(len(v))
    ax.bar(x, v, color=colors, width=0.6, linewidth=0, zorder=3)
    ax.axhline(0, color=T.INK, linewidth=0.8, zorder=2)
    pad = (np.nanmax(np.abs(v)) or 1) * 0.04
    for xi, vi in zip(x, v):
        if np.isfinite(vi):
            ax.text(xi, vi + (pad if vi >= 0 else -pad), fmt.format(vi), ha="center",
                    va="bottom" if vi >= 0 else "top", fontsize=7, color=T.TEXT, zorder=4)
    ax.set_xticks(x, [str(i) for i in values.index])
    ax.margins(x=0.04, y=0.16)
    _style(ax, ylabel, xlabel=xlabel)
    ax.grid(False, axis="x")
    return _img(fig, ylabel)


def category_lines(df: pd.DataFrame, ylabel: str, colors: list[str], width=6.4, height=2.5, xlabel: str = "Spread"):
    """Lines with points over spreads (actual vs predicted)."""
    fig, ax = _new(width, height)
    x = np.arange(len(df.index))
    for i, c in enumerate(df.columns):
        ax.plot(x, df[c].to_numpy(float), color=colors[i % len(colors)], linewidth=1.6, marker="o", markersize=3.5,
                label=str(c), zorder=3)
    ax.set_xticks(x, [str(i) for i in df.index])
    ax.set_xlim(-0.3, len(x) - 0.7)
    _style(ax, ylabel, xlabel=xlabel)
    _legend(ax)
    return _img(fig, ylabel)


def cumulative(df: pd.DataFrame, ylabel: str, colors: list[str], width=13.0, height=2.4):
    """Cumulative return by month, points on each month, endpoint flags (Viz.line)."""
    fig, ax = _new(width, height)
    x = np.arange(len(df.index))
    for i, c in enumerate(df.columns):
        col = colors[i % len(colors)]
        y = df[c].to_numpy(float)
        ax.plot(x, y, color=col, linewidth=1.6, marker="o", markersize=2.8, label=str(c), zorder=3)
        if len(y) and np.isfinite(y[-1]):
            _flag(fig, ax, len(x) - 1 + max(0.25, len(x) * 0.012), y[-1], f"{y[-1]:.1f}%", col)
    ax.axhline(0, color=T.INK, linewidth=0.6, linestyle=":", zorder=2)
    _cat_ticks(ax, x, df.index)
    ax.set_xlim(-0.6, len(x) - 0.4 + max(1.2, len(x) * 0.06))
    _style(ax, ylabel, yfmt="{:.0f}%")
    _legend(ax)
    return _img(fig, ylabel)


def monthly_vs(df: pd.DataFrame, colors: list[str], width=13.0, height=3.1):
    """Monthly returns Model vs SPX on top, Model − SPX underneath, one shared month axis
    (the old app's vconcat of the two bar charts)."""
    fig = Figure(figsize=(width, height))
    FigureCanvasAgg(fig)
    top, bot = fig.subplots(2, 1, sharex=True, gridspec_kw={"height_ratios": [1.7, 1], "hspace": 0.12})
    n, k = len(df), 2
    x = np.arange(n)
    w = 0.8 / k
    for ax in (top, bot):
        ax.set_facecolor(T.PLOT)
        ax.set_axisbelow(True)
        ax.grid(True, color=T.GRID, linewidth=0.7)
    for i, c in enumerate(["Model", "SPX"]):
        top.bar(x + (i - 0.5) * w, df[c].to_numpy(float), width=w, color=colors[i], label=c, linewidth=0, zorder=3)
    diff = (df["Model"] - df["SPX"]).to_numpy(float)
    bot.bar(x, diff, color=[T.GREEN if v > 0 else T.RED for v in diff], width=0.66, linewidth=0, zorder=3,
            label="Model − SPX")
    for ax, lab in ((top, "Return (%)"), (bot, "Model − SPX")):
        ax.axhline(0, color=T.INK, linewidth=0.8, zorder=2)
        _style(ax, lab, yfmt="{:.0f}%")
        ax.grid(False, axis="x")
    top.yaxis.set_major_locator(MaxNLocator(nbins=6))
    bot.yaxis.set_major_locator(MaxNLocator(nbins=4))
    top.tick_params(axis="x", which="both", length=0)
    top.spines["bottom"].set_visible(False)
    _cat_ticks(bot, x, df.index)
    _legend(top)
    return _img(fig, "Monthly returns")


def timeseries(df: pd.DataFrame, colors: list[str], width=13.0, height=1.6, fmt: str = "{:.2f}",
               legend: bool = False, fill: bool = False, yfmt: str | None = None, ylabel: str | None = None):
    """Dated lines with Viz endpoint flags (Viz.line)."""
    import matplotlib.dates as mdates
    fig, ax = _new(width, height)
    idx = pd.to_datetime(df.index)
    x_end = idx.max() + (idx.max() - idx.min()) * 0.03
    for i, c in enumerate(df.columns):
        col = colors[i % len(colors)]
        s = pd.Series(df[c].to_numpy(float), index=idx).dropna()
        ax.plot(s.index, s.values, color=col, linewidth=1.5, label=str(c), zorder=3)
        if fill:
            ax.fill_between(s.index, 0, s.values, color=col, alpha=0.18, linewidth=0, zorder=2)
        _flag(fig, ax, x_end, s.iloc[-1], fmt.format(s.iloc[-1]), col)
    ax.axhline(0, color=T.INK, linewidth=0.6, linestyle=":", zorder=2)
    loc = mdates.AutoDateLocator(minticks=6, maxticks=12)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    ax.xaxis.get_offset_text().set_visible(False)
    ax.set_xlim(idx.min(), x_end + (idx.max() - idx.min()) * 0.05)
    _style(ax, ylabel, yfmt)
    ax.xaxis.set_minor_locator(NullLocator())
    if legend:
        _legend(ax)
    return _img(fig, "time series")
