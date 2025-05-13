import threading
import time
from datetime import datetime
import pandas as pd
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract, ComboLeg
import sys
from tabulate import tabulate, tabulate_formats
import pyarrow  # Ensure pyarrow is installed for Parquet writing
from dateutil.relativedelta import relativedelta

PARQUET_FILE = "spreads.parquet"

class BApi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.contract_details = {}
        self.data = {}
        self.req_id = 1
        self.done = threading.Event()
        self.lock = threading.Lock()
        self.contract_fetch_failed = False  # flag for contract fetch errors

    def error(self, reqId, errorCode, errorString):
        print(f"[Error {reqId}] Code: {errorCode} | Msg: {errorString}")
        if errorCode == 200:
            self.contract_fetch_failed = True

    def nextValidId(self, orderId):
        self.req_id = orderId

    def contractDetails(self, reqId, contractDetails):
        contract = contractDetails.contract
        self.contract_details[contract.conId] = {
            "conId": contract.conId,
            "localSymbol": contract.localSymbol,
            "month": contract.lastTradeDateOrContractMonth
        }

    def contractDetailsEnd(self, reqId):
        print(f"✅ Contract details fetched: {len(self.contract_details)}")
        if len(self.contract_details) >= 8:
            self.done.set()

    def tickPrice(self, reqId, tickType, price, attrib):
        with self.lock:
            if reqId not in self.data:
                self.data[reqId] = {"Bid Size": 0, "Bid Price": 0, "Ask Price": 0, "Ask Size": 0}
            if tickType == 1:  # Bid Price
                self.data[reqId]["Bid Price"] = price
            elif tickType == 2:  # Ask Price
                self.data[reqId]["Ask Price"] = price
            self.data[reqId]["Last Update"] = datetime.now().strftime("%H:%M:%S")

    def tickSize(self, reqId, tickType, size):
        with self.lock:
            if reqId not in self.data:
                self.data[reqId] = {"Bid Size": 0, "Bid Price": 0, "Ask Price": 0, "Ask Size": 0}
            if tickType == 0:  # Bid Size
                self.data[reqId]["Bid Size"] = size
            elif tickType == 3:  # Ask Size
                self.data[reqId]["Ask Size"] = size
            self.data[reqId]["Last Update"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def vix_contract(month_str):
    contract = Contract()
    contract.symbol = "VIX"
    contract.tradingClass = "VX"
    contract.secType = "FUT"
    contract.exchange = "CFE"
    contract.currency = "USD"
    contract.lastTradeDateOrContractMonth = month_str
    return contract


def get_contracts(app):
    def try_fetch(start_month_offset):
        app.contract_details = {}
        app.contract_fetch_failed = False
        app.done.clear()
        contracts = []
        for i in range(8):
            month = (datetime.today() + relativedelta(months=start_month_offset + i)).strftime("%Y%m")
            contract = vix_contract(month)
            contracts.append(contract)
            app.reqContractDetails(app.req_id, contract)
            app.req_id += 1
        app.done.wait(timeout=10)
        return contracts

    # Try from current month
    try_fetch(0)

    if app.contract_fetch_failed or len(app.contract_details) < 8:
        print("⚠️ Retrying with next month as start...")
        try_fetch(1)

    return list(app.contract_details.values())


def spread_contract(leg1, leg2):
    spread = Contract()
    spread.symbol = "VIX"
    spread.secType = "BAG"
    spread.currency = "USD"
    spread.exchange = "CFE"

    l1 = ComboLeg()
    l1.conId = leg1["conId"]
    l1.ratio = 1
    l1.action = "SELL"
    l1.exchange = "CFE"

    l2 = ComboLeg()
    l2.conId = leg2["conId"]
    l2.ratio = 1
    l2.action = "BUY"
    l2.exchange = "CFE"

    spread.comboLegs = [l1, l2]
    return spread


def stream_spread(app, leg1, leg2, req_id, spread_key):
    spread = spread_contract(leg1, leg2)
    app.reqMktData(req_id, spread, "", False, False, [])
    app.data[req_id] = {
        "Spread": spread_key,
        "Contracts": f"{leg1['localSymbol']}/{leg2['localSymbol']}",
        "Months": f"{leg1['month']}/{leg2['month']}",
        "Bid Size": 0,
        "Bid Price": 0,
        "Ask Price": 0,
        "Ask Size": 0,
        "Last Update": ""
    }


def display_loop(app):
    """
    Continuously print the data to console (optional) and
    save the data to a Parquet file for Streamlit consumption.
    """
    while True:
        time.sleep(1)
        with app.lock:
            if app.data:
                df = pd.DataFrame.from_dict(app.data, orient="index")
                df = df[
                    ["Spread", "Contracts", "Months", "Bid Size", "Bid Price", "Ask Price", "Ask Size", "Last Update"]
                ].sort_values("Spread").set_index("Spread")

                # Print to console in place (optional):
                sys.stdout.write("\033c")  # Clear screen
                print("Live Spreads:\n")
                print(tabulate(df, headers="keys", tablefmt=tabulate_formats[4]))
                print(f"Last Update: {df['Last Update'].max()}")

                # Save to parquet
                df.reset_index().to_parquet(PARQUET_FILE, index=False)


def run():
    print("\nStarting multi-spread streaming... (file-based approach)")
    app = BApi()
    app.connect("127.0.0.1", 4001, clientId=104)

    thread = threading.Thread(target=app.run, daemon=True)
    thread.start()
    time.sleep(1)

    details = get_contracts(app)
    if len(details) < 8:
        print("❌ Not enough contracts fetched.")
        app.disconnect()
        return

    # For each consecutive pair, set up the spread
    for i in range(7):
        leg1 = details[i]
        leg2 = details[i + 1]
        spread_key = f"{i + 1}-{i + 2}"
        stream_spread(app, leg1, leg2, app.req_id, spread_key)
        app.req_id += 1

    display_loop(app)


if __name__ == "__main__":
    run()
