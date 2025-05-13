import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import xml.etree.ElementTree as ET
import requests
from datetime import datetime
import subprocess
import altair as alt
import utils

from prep_data_2 import load_data, feature_engineer
from backtest_lin_ls import load_models, generate_all_predictions, ranked_strategy_vol_adjusted

# If your IBKR spread streamer writes updated data to this file:
SPREAD_FILE = "sheet/spreads.parquet"

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

# ------------------ Constants ------------------ #
strategy_start_date = datetime(2024, 8, 27)
chuck_initial_investment = 7000
chuck_strike = 20699.76
chuck_pct = chuck_initial_investment / chuck_strike
chuck_invest_date = datetime(2024, 3, 6)

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

    # 2) Optional Refresh
    if st.button("Refresh Spread File"):
        try:
            st.session_state["df_spread"] = load_spread_parquet()
        except Exception as e:
            st.error(f"Error reading {SPREAD_FILE}: {e}")

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
            cmd = [
                "conda", "run", "-n", "algo",
                "bash", "-c",
                "python update_data.py"
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            st.success("Data updated successfully!")
        else:
            st.error(f"Data update failed with return code {result.returncode}")
            if result.stderr:
                st.write("**Script errors:**")
                st.code(result.stderr)

        if result.stdout:
            st.write("**Script output:**")
            st.code(result.stdout)

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
                        resid = signal[f"Spread_{spread}"].values[0] - signal[f"Pred_{spread}"].values[0]
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

    token = "154979803551046183567560"
    queryId = "1155973"
    base_url = "https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService"

    if st.button("Refresh NAV Data"):
        utils.git_push("auto push on NAV ref")
        status_placeholder = st.empty()
        progress_bar = st.progress(0)
        try:
            status_placeholder.info("Requesting Flex Statement Reference Code...")
            send_url = f"{base_url}/SendRequest?t={token}&q={queryId}&v=3"
            send_response = requests.get(send_url)
            root = ET.fromstring(send_response.text)
            ref_code = root.find("ReferenceCode").text
            status_placeholder.success(f"Received Reference Code: {ref_code}")

            get_url = f"{base_url}/GetStatement?t={token}&q={ref_code}&v=3"
            for attempt in range(10):
                progress_bar.progress((attempt + 1) / 10)
                status_placeholder.info(f"Attempt {attempt + 1}: Requesting Flex Statement...")
                get_response = requests.get(get_url)
                if get_response.content.strip():
                    with open("sheet/Summ.xml", "wb") as f:
                        f.write(get_response.content)
                    status_placeholder.success("Flex Statement downloaded successfully.")
                    time.sleep(3)
                    break
                status_placeholder.warning(f"Statement not ready. Waiting 10 seconds...")
                time.sleep(10)
            else:
                status_placeholder.error("All attempts failed.")
            progress_bar.empty()
        except Exception as e:
            status_placeholder.error(f"Error: {e}")
            progress_bar.empty()

    if not os.path.exists("sheet/Summ.xml"):
        st.warning("Summ.xml not found. Please refresh NAV data.")
    else:
        try:
            tree = ET.parse("sheet/Summ.xml")
            root = tree.getroot()

            rows = []
            for statement in root.iter("FlexStatement"):
                date = statement.get("fromDate")
                change_nav = statement.find("ChangeInNAV")
                start_nav = float(change_nav.get("startingValue", 0))
                net_inflow = float(change_nav.get("depositsWithdrawals", 0))
                end_nav = float(change_nav.get("endingValue", 0))
                rows.append([date, start_nav, net_inflow, end_nav])

            df = pd.DataFrame(rows, columns=["Date", "Start NAV", "Net Inflow", "End NAV"])

            total_inflows = df["Net Inflow"].sum()
            benjin_investment = total_inflows - chuck_initial_investment

            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            df = df.dropna(subset=["Date", "Start NAV", "End NAV"]).sort_values("Date")
            df = df[df["Start NAV"] >= 100]
            df = df[df["Date"] >= strategy_start_date]

            if df.empty:
                st.error("No valid NAV data after filtering. Check Summ.xml.")
                st.stop()

            # Calculate Returns
            df["Cumulative NAV"] = df["End NAV"]

            df["Ben NAV"] = df["Cumulative NAV"] * (1 - chuck_pct)
            df["Chuck NAV"] = df["Cumulative NAV"] * chuck_pct

            df_monthly = df.set_index("Date").resample("ME").agg({
                "Start NAV": "first",
                "End NAV": "last",
                "Net Inflow": "sum"
            })
            df_monthly["MonthlyReturn"] = (
                (df_monthly["End NAV"] - df_monthly["Start NAV"] - df_monthly["Net Inflow"]) / df_monthly["Start NAV"]
            ) * 100
            df_monthly.index = df_monthly.index.strftime("%b %y")

            df_yearly = df.set_index("Date").resample("YE").agg({
                "Start NAV": "first",
                "End NAV": "last",
                "Net Inflow": "sum"
            })
            df_yearly["YearlyReturn"] = (
                (df_yearly["End NAV"] - df_yearly["Start NAV"] - df_yearly["Net Inflow"]) / df_yearly["Start NAV"]
            ) * 100
            df_yearly.index = df_yearly.index.year

            st.subheader("Returns")
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

            st.subheader("Performance Chart")
            st.line_chart(
                df.set_index("Date")[["Cumulative NAV"]].rename(columns={"Cumulative NAV": "NAV Trajectory"})
            )

            # Combine Benjin + Wagon logic here.
            # Benjin next to the header with an inline image:
            st.markdown(
                """
                <div style="display: flex; align-items: center; gap: 10px; margin-top: 1em;">
                    <img src="https://cdn.freebiesupply.com/images/large/2x/washington-redskins-logo-transparent.png" style="height:50px; width:auto;"/>
                </div>
                """,
                unsafe_allow_html=True
            )
            df_ben = df.copy()
            ben_current_value = df_ben["Ben NAV"].iloc[-1]
            ben_pnl = ben_current_value - benjin_investment
            ben_return = (ben_pnl / benjin_investment) * 100
            ben_data = pd.DataFrame({
                "Initial Investment": [benjin_investment],
                "Current Value": [ben_current_value],
                "PnL": [ben_pnl],
                "Returns %": [ben_return]
            }, index=["Benjin"])
            st.table(
                ben_data.style.map(
                    lambda v: "color: green" if v > 0 else "color: red", subset=["PnL", "Returns %"]
                ).format("{:.2f}")
            )

            # Wagon next to the header with an inline image:
            st.markdown(
                """
                <div style="display: flex; align-items: center; gap: 10px; margin-top: 2em;">
                    <img src="https://upload.wikimedia.org/wikipedia/commons/archive/5/5c/20240330094136%21Chicago_Bears_logo.svg" style="height:40px; width:auto;"/>
                </div>
                """,
                unsafe_allow_html=True
            )
            wagon_df = df[df["Date"] >= pd.to_datetime(chuck_invest_date)].copy()
            chuck_current_value = wagon_df["Chuck NAV"].iloc[-1]
            wagon_pnl = chuck_current_value - chuck_initial_investment
            wagon_return = (wagon_pnl / chuck_initial_investment) * 100
            chuck_data = pd.DataFrame({
                "Initial Investment": [chuck_initial_investment],
                "Current Value": [chuck_current_value],
                "PnL": [wagon_pnl],
                "Returns %": [wagon_return]
            }, index=["Chuck"])
            st.table(
                chuck_data.style.map(
                    lambda v: "color: green" if v > 0 else "color: red", subset=["PnL", "Returns %"]
                ).format("{:.2f}")
            )

            st.session_state["df"] = df  # Keep for reference if needed.

        except Exception as e:
            st.error(f"Error processing Summ.xml: {e}")
