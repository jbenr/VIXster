import threading
import time
import pandas as pd
from datetime import datetime
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
import pyarrow
import pyarrow.parquet as pq

SPX_HISTORY_FILE = "spx_history.parquet"

class SPXHistApi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        # We'll store historical bars here
        self.historical_data = []
        # We'll track how many historical requests remain
        self.pending_req = 0
        self.lock = threading.Lock()
        self.done = threading.Event()

    def error(self, reqId, errorCode, errorString):
        print(f"[Error {reqId}] Code: {errorCode} | Msg: {errorString}")

    def nextValidId(self, orderId):
        print(f"Next valid ID is {orderId}")
        # We can start requests after we have a valid ID

    def historicalData(self, reqId, bar):
        """
        Called for each bar in the historical data response.
        'bar' is an IB 'BarData' object:
            bar.date, bar.open, bar.high, bar.low, bar.close,
            bar.volume, bar.barCount, bar.wap
        """
        with self.lock:
            self.historical_data.append({
                "date": bar.date,
                "open": bar.open,
                "high": bar.high,
                "low": bar.low,
                "close": bar.close,
                "volume": bar.volume,
                "barCount": bar.barCount,
                "wap": bar.wap
            })

    def historicalDataEnd(self, reqId, start, end):
        """
        Called once the entire historical dataset is finished.
        """
        print(f"Finished historical request for ReqId={reqId}, start={start}, end={end}")
        with self.lock:
            self.pending_req -= 1
            if self.pending_req <= 0:
                print("All historical data requests completed.")
                self.done.set()


def spx_index_contract() -> Contract:
    """
    Returns a Contract object representing the S&P 500 Index (SPX).
    IBKR 'secType' for an index is typically 'IND'.
    The exchange for SPX is often 'CBOE' or 'CBOEIND'.
    """
    contract = Contract()
    contract.symbol = "SPX"         # IB sometimes uses "SPX" for the S&P 500 Index
    contract.secType = "IND"        # IND means index
    contract.currency = "USD"
    contract.exchange = "CBOE"      # or "CBOEIND" sometimes. Possibly "SMART" for routing.

    return contract


def request_spx_history(app: SPXHistApi):
    """
    Request 1 day of 5-minute bars for the SPX index from IBKR.
    Customize 'durationStr', 'barSizeSetting', or 'whatToShow' as you like.
    """
    req_id = 1
    app.pending_req += 1

    contract = spx_index_contract()

    # For details on these parameters, see IB API docs for reqHistoricalData
    app.reqHistoricalData(
        reqId=req_id,
        contract=contract,
        endDateTime="",            # "" -> current time
        durationStr="1 D",         # can do "1 D", "1 W", "1 M", "6 M", "1 Y", etc.
        barSizeSetting="5 mins",   # "1 min", "5 mins", "15 mins", "30 mins", "1 hour" ...
        whatToShow="MIDPOINT",     # or "TRADES", "BID", "ASK" ...
        useRTH=0,                  # 0 = all data, 1 = regular trading hours only
        formatDate=1,              # 1 => string dates, 2 => long int
        keepUpToDate=False,
        chartOptions=[]
    )


def run_spx_history():
    print("Starting SPX historical data fetch...")

    # 1) Create the IBKR app
    app = SPXHistApi()
    app.connect("127.0.0.1", 4001, clientId=5566)  # Adjust clientId if needed

    # 2) Launch a background thread to run the socket
    t = threading.Thread(target=app.run, daemon=True)
    t.start()

    time.sleep(1)  # let the connection settle

    # 3) Request the SPX historical data
    request_spx_history(app)

    # 4) Wait for data to finish
    app.done.wait(timeout=60)  # wait up to 60 seconds

    # 5) Convert the collected bars to a DataFrame
    df = pd.DataFrame(app.historical_data)
    if not df.empty:
        # Sort by date (optional)
        # IBKR date is typically "20230403  09:30:00", so we might parse it
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Save to Parquet
        df.to_parquet(SPX_HISTORY_FILE, index=False)
        print(f"Saved {len(df)} bars to {SPX_HISTORY_FILE}.")
        print(df.tail())
    else:
        print("No historical bars received for SPX. Possibly no data or a request issue.")

    # 6) Disconnect
    app.disconnect()
    print("All done.")


if __name__ == "__main__":
    run_spx_history()
