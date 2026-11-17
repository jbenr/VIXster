"""VIXster 2.0 — Dash app in the Viz theme. Run: `vixster app` (dev) or `vixster serve` (gunicorn)."""
from __future__ import annotations

from dash import Dash, Input, Output, dcc, html

from vixster.app.pages import curve, data, model, performance

PAGES = {"data": ("Data", data), "model": ("Model", model), "performance": ("Performance", performance),
         "curve": ("Curve PCA", curve)}

app = Dash(
    __name__,
    title="The Model",
    update_title=None,
    suppress_callback_exceptions=True,
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"},
               {"name": "theme-color", "content": "#0D1014"}],
    external_stylesheets=["https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"],
)
server = app.server

# Every page is in the DOM from the start and the tabs just show/hide them,
# like Streamlit's st.tabs: switching is instant and nothing reruns.
app.layout = html.Div([
    dcc.Location(id="url"),
    html.Div(html.H1("The Model"), className="masthead"),
    dcc.Tabs(id="tabs", value="data", className="folder-tabs", parent_className="folder-tabs-parent",
             mobile_breakpoint=0,
             children=[dcc.Tab(label=lbl, value=key, className="folder-tab", selected_className="folder-tab--open")
                       for key, (lbl, _) in PAGES.items()]),
    *[html.Div(mod.layout(), id=f"page-{key}", className="page",
               style=None if key == "data" else {"display": "none"}) for key, (_, mod) in PAGES.items()],
], className="shell")

app.clientside_callback(
    """
    function(tab, search) {
        const ctx = window.dash_clientside.callback_context;
        const fromUrl = new URLSearchParams(search || "").get("tab");
        const keys = %s;
        if (ctx.triggered.length && ctx.triggered[0].prop_id === "url.search" && keys.includes(fromUrl)) tab = fromUrl;
        return [tab, ...keys.map(k => k === tab ? {} : {display: "none"})];
    }
    """ % list(PAGES),
    Output("tabs", "value"),
    *[Output(f"page-{k}", "style") for k in PAGES],
    Input("tabs", "value"),
    Input("url", "search"),
)

for _, mod in PAGES.values():
    mod.register(app)


def run(host: str = "0.0.0.0", port: int = 8050, debug: bool = False):
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    run(debug=True)
