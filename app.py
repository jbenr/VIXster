import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
import subprocess
import altair as alt
import utils

from prep_data_2 import load_data, feature_engineer
from backtest_lin_ls import load_models, generate_all_predictions, ranked_strategy_vol_adjusted
from update_data import pull_performance
import update_data

# If your IBKR spread streamer writes updated data to this file:
SPREAD_FILE = "data/spreads.parquet"

st.set_page_config(layout="wide")
st.markdown(
    """
    <style>
    table {
        width: auto !important;
        table-layout: auto !important;
        margin-left: auto !important;
        margin-right: auto !important;
        border-collapse: collapse !important;
    }

    .element-container:has(table) {
        display: flex;
    }

    thead th, tbody td {
        white-space: nowrap !important;
        padding: 6px 12px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("The Model")

# ------------------ Tabs ------------------ #
tabs = st.tabs(["Data", "Model", "Performance"])

# ------------------ DATA TAB (File-based Refresh) ------------------ #
with tabs[0]:
    st.header("Live Spread Quotes")

    def parse_month_code(code_8digits):
        yyyy = code_8digits[:4]
        mm = code_8digits[4:6]
        month_map = {
            '01': 'Jan','02': 'Feb','03': 'Mar','04': 'Apr','05': 'May','06': 'Jun',
            '07': 'Jul','08': 'Aug','09': 'Sep','10': 'Oct','11': 'Nov','12': 'Dec'
        }
        month_abbr = month_map.get(mm, mm)
        yy = yyyy[-2:]
        return f"{month_abbr}{yy}"

    def parse_months(m_str):
        """
        e.g. '20250416/20250521' -> 'Apr25/May25'
        ignoring the day portion
        """
        left, right = m_str.split("/")
        left_code = left[:6]
        right_code = right[:6]
        return f"{parse_month_code(left_code)}/{parse_month_code(right_code)}"

    def load_spread_parquet():
        """Load the parquet file and parse the Months column if present."""
        if not os.path.exists(SPREAD_FILE):
            st.warning(f"No spread data file found at {SPREAD_FILE}. Waiting for streamer to create it.")
            st.stop()

        df_spread = pd.read_parquet(SPREAD_FILE)
        if "Spread" in df_spread.columns:
            df_spread = df_spread.set_index("Spread")

        if "Months" in df_spread.columns:
            df_spread["Months"] = df_spread["Months"].apply(parse_months)

        return df_spread

    # 1) Automatic load on first page load
    if "df_spread" not in st.session_state:
        try:
            st.session_state["df_spread"] = load_spread_parquet()
        except Exception as e:
            st.error(f"Error reading {SPREAD_FILE}: {e}")
            st.stop()

    # Layout with columns
    col1, col2, spacer = st.columns([1, 1, 2])

    # 1) Refresh Spread File
    with col1:
        if st.button("Refresh Spread File"):
            try:
                st.session_state["df_spread"] = load_spread_parquet()
            except Exception as e:
                st.error(f"Error reading {SPREAD_FILE}: {e}")

    # 2) IBKR Login Button
    with col2:
        if st.button("IBKR Login"):
            try:
                result = subprocess.run(
                    ["bash", "/home/han/pod/VIXster/launch_ibkr.sh"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                # st.success("IBKR Gateway launched successfully!")
                st.toast(result.stdout)
            except subprocess.CalledProcessError as e:
                st.error("Failed to launch IBKR Gateway.")
                st.text(e.stderr)

    # 3) Display the table (if loaded)
    if "df_spread" in st.session_state and st.session_state["df_spread"] is not None:
        df_spread = st.session_state["df_spread"]
        styled = (
            df_spread
            .style
            .format(subset=["Bid Price", "Ask Price"], formatter="{:.2f}")
        )
        st.table(styled)
        if "Last Update" in df_spread.columns:
            st.markdown(f"Last Update: {df_spread['Last Update'].max()}")
    else:
        st.info("No spread data loaded yet.")

    st.markdown("<hr style='margin-top: 1em; margin-bottom: 1em;'>", unsafe_allow_html=True)
    # 4) Historical data
    col1, col2 = st.columns(2)

    with col1:
        HIST_FILE = "data/vix_spreads.parquet"
        st.write("### Historical VIX Spreads")
        if os.path.exists(HIST_FILE):
            df_hist = pd.read_parquet(HIST_FILE)
            if "Date" in df_hist.columns:
                df_hist["Date"] = pd.to_datetime(df_hist["Date"], errors="coerce")
                df_hist.sort_values("Date", inplace=True)
            st.dataframe(df_hist[['1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8']].tail(10)
                         .style
                         .format(formatter="{:.2f}"),
                         use_container_width=True)

            df_hist.index = pd.to_datetime(df_hist.index)
            last_update_hist = df_hist.index.max()
            if pd.Timestamp.now() - last_update_hist >= pd.Timedelta("2 days"):
                st.markdown(
                    f"<span style='color:red;'>Last Update: {last_update_hist.strftime('%Y-%m-%d')}</span>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(f"Last Update: {last_update_hist.strftime('%Y-%m-%d')}")
        else:
            st.info(f"No historical spread file found at {HIST_FILE}.")

    with col2:
        FEAT_FILE = "data/fred.parquet"
        st.write("### Historical Feature Data")
        if os.path.exists(FEAT_FILE):
            try:
                df_feat = pd.read_parquet(FEAT_FILE)
                # If there's a date col, parse + sort
                if "Date" in df_feat.columns:
                    df_feat["Date"] = pd.to_datetime(df_feat["Date"], errors="coerce")
                    df_feat.sort_values("Date", inplace=True)
                st.dataframe(df_feat.tail(10), use_container_width=True)

                df_feat.index = pd.to_datetime(df_feat.index)
                last_update_feat = df_feat.index.max()
                if pd.Timestamp.now() - last_update_feat >= pd.Timedelta("2 days"):
                    st.markdown(
                        f"<span style='color:red;'>Last Update: {last_update_feat.strftime('%Y-%m-%d')}</span>",
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(f"Last Update: {last_update_feat.strftime('%Y-%m-%d')}")
            except Exception as ex:
                st.error(f"Error reading {FEAT_FILE}: {ex}")
        else:
            st.info(f"No feature data file found at {FEAT_FILE}.")

    if st.button("Update Historical Data"):
        # Show a spinner message while the script runs
        with st.spinner("Running update_data.py ..."):
            try:
                # Directly call the run() function from update_data.py
                update_data.run()
                st.success("Data updated successfully!")
            except Exception as e:
                st.error(f"Data update failed with error:\n{e}")


# ------------------ MODEL TAB ------------------ #
with tabs[1]:
    # st.header("Modelo")
    st.markdown(
        """
        <div style="display: flex;">
            <h2>Modelo</h2>
            <img src="https://upload.wikimedia.org/wikipedia/commons/5/56/Cerveza-Modelo.png"
                 style="height:90px; width:auto; filter: invert(1);" />
        </div>
        """,
        unsafe_allow_html=True
    )

    def compute_contracts(weight_long, weight_short, tol=0.1):

        if weight_long < weight_short:
            r = weight_short / weight_long
            x = 1
            while True:
                candidate = r * x
                rounded = round(candidate)
                error = abs(candidate - rounded)
                if error <= tol:
                    return (x, rounded)  # (long_contracts, short_contracts)
                x += 1
        else:
            r = weight_long / weight_short
            x = 1
            while True:
                candidate = r * x
                rounded = round(candidate)
                error = abs(candidate - rounded)
                if error <= tol:
                    return (rounded, x)  # (long_contracts, short_contracts)
                x += 1

    # Button to compute live signal.
    if st.button("Run Modelo"):
        try:
            try:
                df = load_data(live=True)
            except Exception as e:
                st.error(e)
                df = load_data()

            utils.pdf(df.tail(10))
            df_data = feature_engineer(df)

            target_spreads = [f"{i}-{i + 1}" for i in range(1, 8)][1:]
            model_paths = {spread: f"data/models/linear_model_{spread}.pkl" for spread in target_spreads}

            best_feats = pd.read_parquet('data/optimal_models.parquet')
            maximizing_by = 'DirectionalAcc'
            feature_cols_dict = {
                spread: list(best_feats[best_feats['Spread'] == spread].loc[
                                 best_feats[best_feats['Spread'] == spread][maximizing_by].idxmax()
                             ]['SubsetBlocks'])
                for spread in target_spreads
            }
            models = load_models(model_paths)

            df_preds = generate_all_predictions(models, df_data, feature_cols_dict)
            df_result = ranked_strategy_vol_adjusted(
                df_preds, target_spreads, vol_lookback=60, z_lookback=30, z_thresh=1.0, exit_z_thresh=0.25)

            signal = df_result.tail(1)
            signal_date = signal.index[0] if hasattr(signal.index, '__getitem__') else None

            if signal is not None:
                if signal_date is not None:
                    st.subheader(f"Signal {signal_date.strftime('%Y-%m-%d')}")
                else:
                    st.subheader("Signal")
                output_rows = []

                # Check if a trade is signaled.
                if signal["Trade"].values[0] in [True, "TRUE", "True"]:
                    weight_long = signal["Weight_Long"].values[0]
                    weight_short = signal["Weight_Short"].values[0]
                    # Scale such that the smaller weight becomes 1 contract.
                    long_contracts, short_contracts = compute_contracts(weight_long, weight_short, tol=0.1)
                    # Also capture the designated long and short spreads.
                    long_spread = signal["LongSpread"].values[0]
                    short_spread = signal["ShortSpread"].values[0]
                else:
                    # No trade: nothing to scale.
                    long_contracts = None
                    short_contracts = None
                    long_spread = None
                    short_spread = None

                output_dict = {
                    "Actual": {},
                    "Pred": {},
                    "Resid": {},
                    "Z": {},
                    "Signal": {},
                    "Contracts": {}
                }

                for spread in target_spreads:
                    if signal["Trade"].values[0] in [True, "TRUE", "True"]:
                        if spread == long_spread:
                            output_dict["Signal"][spread] = "Long"
                            output_dict["Contracts"][spread] = round(long_contracts, 2)
                        elif spread == short_spread:
                            output_dict["Signal"][spread] = "Short"
                            output_dict["Contracts"][spread] = round(short_contracts, 2)
                        else:
                            output_dict["Signal"][spread] = ""
                            output_dict["Contracts"][spread] = ""
                    else:
                        output_dict["Signal"][spread] = ""
                        output_dict["Contracts"][spread] = ""

                    actual = signal[f"Spread_{spread}"].values[0]
                    pred = signal[f"Pred_{spread}"].values[0]
                    z = signal[f"Z_{spread}"].values[0]
                    resid = pred - actual

                    output_dict["Actual"][spread] = round(actual, 2)
                    output_dict["Pred"][spread] = round(pred, 2)
                    output_dict["Resid"][spread] = round(resid, 2)
                    output_dict["Z"][spread] = round(z, 2)

                df_display = pd.DataFrame(output_dict)
                df_display = df_display.T  # Now rows are: Signal, Contracts, etc.

                def highlight_signal_columns(col):
                    signal = df_display.loc["Signal", col.name]
                    if signal == "Long":
                        return ["background-color: #154734"] * len(col)  # Pale Green
                    elif signal == "Short":
                        return ["background-color: #9e1b32"] * len(col)  # Light Coral
                    else:
                        return [""] * len(col)


                df_display = (
                    df_display
                    .map(lambda x: f"{x:.2f}"
                    if isinstance(x, (int, float, np.integer, np.floating, np.number))
                    else str(x))
                    .astype(str)
                )

                styled_df = df_display.style.apply(highlight_signal_columns, axis=0)
                st.table(styled_df)

                # Line Chart: Actual vs. Predicted Spread Values
                data = {"Spread": [], "Type": [], "Value": []}
                for spread in target_spreads:
                    # Make sure both columns exist in signal
                    actual_col = f"Spread_{spread}"
                    pred_col = f"Pred_{spread}"
                    if actual_col in signal.columns and pred_col in signal.columns:
                        actual_value = signal[actual_col].values[0]
                        predicted_value = signal[pred_col].values[0]
                        data["Spread"].extend([spread, spread])
                        data["Type"].extend(["Actual", "Predicted"])
                        data["Value"].extend([actual_value, predicted_value])
                    else:
                        st.warning(f"Columns for spread {spread} are not available in signal.")

                df_plot = pd.DataFrame(data)

                chart = alt.Chart(df_plot).mark_line(point=True).encode(
                    x=alt.X("Spread:N", scale=alt.Scale(domain=target_spreads), title="Spread"),
                    y=alt.Y("Value:Q", title="Spread Value"),
                    color=alt.Color("Type:N", title="Series")
                ).properties(
                    title="Actual Spreads vs Predicted Spreads",
                    width=800,
                    height=300
                )

                # 2) Bar Chart: Net Residual per Spread (Actual - Prediction for today)
                residuals = []
                long_spread = signal["LongSpread"].values[0] if "LongSpread" in signal.columns else None
                short_spread = signal["ShortSpread"].values[0] if "ShortSpread" in signal.columns else None
                for spread in target_spreads:
                    if f"Spread_{spread}" in signal.columns and f"Pred_{spread}" in signal.columns:
                        resid = signal[f"Pred_{spread}"].values[0] - signal[f"Spread_{spread}"].values[0]
                        # Set color: green if spread == long_spread; red if spread == short_spread; else gray.
                        if spread == long_spread:
                            color = "green"
                        elif spread == short_spread:
                            color = "red"
                        else:
                            color = "gray"
                        residuals.append({"Spread": spread, "Residual": resid, "Color": color})
                residual_df = pd.DataFrame(residuals)

                bar_resid = alt.Chart(residual_df).mark_bar().encode(
                    x=alt.X("Spread:N", title="Spread"),
                    y=alt.Y("Residual:Q", title="Net Residual"),
                    color=alt.Color("Color:N", scale=None, title="Trade Side")
                ).properties(
                    width=500,
                    height=400,
                    title="Net Residual per Spread"
                )

                # CHART 3: Bar Chart for Z-Scores per Spread (Today's Data)
                z_scores = []
                for spread in target_spreads:
                    if f"Z_{spread}" in signal.columns:
                        z_val = signal[f"Z_{spread}"].values[0]
                        if spread == long_spread:
                            color = "green"
                        elif spread == short_spread:
                            color = "red"
                        else:
                            color = "gray"
                        z_scores.append({"Spread": spread, "Z_Score": z_val, "Color": color})
                zscore_df = pd.DataFrame(z_scores)

                bar_z = alt.Chart(zscore_df).mark_bar().encode(
                    x=alt.X("Spread:N", title="Spread"),
                    y=alt.Y("Z_Score:Q", title="Z-Score"),
                    color=alt.Color("Color:N", scale=None, title="Trade Side")
                ).properties(
                    width=500,
                    height=400,
                    title="Z-Scores per Spread"
                )

                st.altair_chart(bar_resid, use_container_width=False)
                st.altair_chart(bar_z, use_container_width=False)
                st.altair_chart(chart, use_container_width=False)

                st.markdown("<hr style='margin-top: 1em; margin-bottom: 1em;'>", unsafe_allow_html=True)

                st.write("Historical Signals:")
                st.dataframe(df_result.tail(50))

                st.write("Latest Data:")
                st.dataframe(df_data.tail(5))
            else:
                st.warning("No signal computed. Please check your data.")
        except Exception as e:
            st.error(f"Error computing live signal: {e}")

# ------------------ PERFORMANCE TAB ------------------ #
with tabs[2]:
    st.header("Historical Performance")

    # -----------------------------
    # 1) “Refresh NAV Data” button
    # -----------------------------
    if st.button("Refresh NAV Data"):
        status = st.empty()
        try:
            # Now saving to "data/performance.parquet"
            df, investor_results = pull_performance(
                out_parquet="data/performance.parquet"
            )
            st.session_state["perf_df"] = df
            st.session_state["investor_results"] = investor_results
            status.success(f"Performance data updated at {datetime.now():%Y-%m-%d %H:%M:%S}.")
            utils.git_push(f"auto push on NAV ref {datetime.now()}")

        except Exception as e:
            status.error(f"Error updating performance data: {e}")

    # ------------------------------------------------------
    # 2) Decide where to get the DataFrame (df) from:
    #    • If session_state has “perf_df”, use that.
    #    • Else if "data/performance.parquet" exists, load df from disk.
    #    • Otherwise stop and ask the user to refresh.
    # ------------------------------------------------------
    if "perf_df" in st.session_state:
        df = st.session_state["perf_df"]
    elif os.path.exists("data/performance.parquet"):
        df = pd.read_parquet("data/performance.parquet")
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)
    else:
        st.warning("No performance data found. Click “Refresh NAV Data” to pull the latest.")
        st.stop()

    # ------------------------------------------------------
    # 3) Decide where to get investor_results from:
    #    • If session_state has “investor_results”, use that.
    #    • Otherwise, leave it empty (user must refresh to populate).
    # ------------------------------------------------------
    investor_results = st.session_state.get("investor_results", {})

    # If df is empty or investor_results was never populated, show an info bar:
    if df.empty:
        st.error("Loaded performance data is empty. Try clicking “Refresh NAV Data” again.")
        st.stop()

    # ─── 4) Strategy‐level Monthly & Yearly Returns ────────────────────────────────
    df_monthly = (
        df.set_index("Date")
          .resample("ME")
          .agg({
              "Start NAV": "first",
              "End NAV":   "last",
              "Net Inflow": "sum"
          })
    )
    df_monthly["MonthlyReturn"] = (
        (df_monthly["End NAV"] - df_monthly["Start NAV"] - df_monthly["Net Inflow"])
        / df_monthly["Start NAV"]
    ) * 100
    df_monthly.index = df_monthly.index.strftime("%b %y")

    df_yearly = (
        df.set_index("Date")
          .resample("YE")
          .agg({
              "Start NAV": "first",
              "End NAV":   "last",
              "Net Inflow": "sum"
          })
    )
    df_yearly["YearlyReturn"] = (
        (df_yearly["End NAV"] - df_yearly["Start NAV"] - df_yearly["Net Inflow"])
        / df_yearly["Start NAV"]
    ) * 100
    df_yearly.index = df_yearly.index.year

    st.subheader("Strategy Returns")
    st.table(
        df_monthly[["MonthlyReturn"]]
        .T
        .style.map(lambda v: "color: green" if v > 0 else "color: red")
        .format("{:.2f}%")
    )
    st.table(
        df_yearly[["YearlyReturn"]]
        .T
        .style.map(lambda v: "color: green" if v > 0 else "color: red")
        .format("{:.2f}%")
    )

    total_return = (
        df["End NAV"].iloc[-1]
        - df["Start NAV"].iloc[0]
        - df["Net Inflow"].sum()
    ) / df["Start NAV"].iloc[0]
    st.table(
        pd.DataFrame({"Total Return %": [total_return * 100]}, index=["Strategy"])
        .style.map(lambda v: "color: green" if v > 0 else "color: red")
        .format("{:.2f}%")
    )

    # ─── 5) Strategy‐level Performance Chart ───────────────────────────────────────
    # st.subheader("Performance Chart (% Return)")
    #
    # # Compute daily return: ((End NAV - Net Inflow) - Start NAV) / Start NAV
    # df["DailyReturn"] = ((df["End NAV"] - df["Net Inflow"]) - df["Start NAV"]) / df["Start NAV"]
    #
    # # Compute cumulative return by compounding: (1 + r).cumprod() - 1
    # df["CumulativeReturn"] = (1 + df["DailyReturn"]).cumprod() - 1
    #
    # st.line_chart(
    #     df.set_index("Date")[["CumulativeReturn"]].rename(columns={"CumulativeReturn": "% Return"})
    # )
    st.markdown("<hr style='margin-top: 1em; margin-bottom: 1em;'>", unsafe_allow_html=True)

    # ─── 7) Investor Details (logo above each table, then separate st.table) ────
    st.subheader("The Gentlemen")
    if investor_results:
        for name, info in investor_results.items():
            logo_url = info.get("logo_url", "")
            logo_height = info.get("logo_height", 40)

            # Render logo via HTML <img> tag
            if logo_url:
                st.markdown(
                    f"""
                    <div style="display: flex; align-items: center; gap: 10px; margin-top: 1em;">
                        <img src="{logo_url}" style="height:{logo_height}px; width:auto;"/>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div style='font-size:1.2em; font-weight:bold; margin-top:1em;'>{name}</div>",
                    unsafe_allow_html=True
                )

            inv_df = pd.DataFrame({
                "Initial Investment": [info["initial_investment"]],
                "Current Value":      [info["current_value"]],
                "PnL":                [info["pnl"]],
                "Returns %":          [info["return_pct"]]
            }, index=[name])

            st.table(
                inv_df.style
                .format({
                    "Initial Investment": "${:,.2f}",
                    "Current Value":      "${:,.2f}",
                    "PnL":                "${:,.2f}",
                    "Returns %":          "{:.2f}%"
                })
                .applymap(lambda v: "color: green" if v > 0 else "color: red", subset=["PnL", "Returns %"])
            )
    else:
        st.info("Investor details not yet populated. Click “Refresh NAV Data” to fetch investor info.")

    # ─── 8) Store df in session_state in case other tabs need it ──────────────────
    st.session_state["df"] = df