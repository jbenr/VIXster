import pandas as pd
import streamlit as st

def render_signal_table(spreads):
    df = pd.DataFrame(spreads)
    df = df[["Spread", "Contracts", "Months", "Bid Size", "Bid Price", "Ask Price", "Ask Size", "Last Update"]]

    def highlight_bid_ask(val, col):
        if col == "Bid Price":
            return "color: green"
        elif col == "Ask Price":
            return "color: red"
        return ""

    styled = df.style.applymap(lambda v: "color: green", subset=["Bid Price"]) \
                     .applymap(lambda v: "color: red", subset=["Ask Price"])
    st.dataframe(styled)
