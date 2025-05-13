from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.common import BarData
import threading
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
from tabulate import tabulate, tabulate_formats


class BApi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.contract_details = {}
        self.data = {}
        self.done = threading.Event()
        self.hist_data_ready = threading.Event()
        self.req_id = 1

    def error(self, reqId, errorCode, errorString):
        print(f"[Error {reqId}] Code: {errorCode} | Msg: {errorString}")

    def nextValidId(self, orderId):
        print(f"Next Valid ID: {orderId}")

    def contractDetails(self, reqId, contractDetails):
        contract = contractDetails.contract
        print(f"Received Contract Detail: {contract.localSymbol} | Exp: {contract.lastTradeDateOrContractMonth}")
        self.contract_details[contract.conId] = {
            "conId": contract.conId,
            "localSymbol": contract.localSymbol,
            "lastTradeDateOrContractMonth": contract.lastTradeDateOrContractMonth,
            "exchange": contract.exchange,
            "currency": contract.currency
        }

    def contractDetailsEnd(self, reqId):
        print(f"Contract Details End received. Total: {len(self.contract_details)}")
        if len(self.contract_details) == self.expected_count:
            self.done.set()

    def tickPrice(self, reqId, tickType, price, attrib):
        if reqId not in self.data:
            self.data[reqId] = {}
        self.data[reqId][f"price_{tickType}"] = price

    def tickSize(self, reqId, tickType, size):
        if reqId not in self.data:
            self.data[reqId] = {}
        self.data[reqId][f"size_{tickType}"] = size

    def historicalData(self, reqId, bar: BarData):
        self.data[reqId] = {
            "datetime": bar.date,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume
        }

    def historicalDataEnd(self, reqId, start, end):
        self.hist_data_ready.set()


def vix_contract(month_str):
    contract = Contract()
    contract.symbol = "VIX"
    contract.tradingClass = "VX"
    contract.secType = "FUT"
    contract.exchange = "CFE"
    contract.currency = "USD"
    contract.lastTradeDateOrContractMonth = month_str
    return contract


def get_contract_details(app, months_ahead=9):
    print(f"\nRequesting contract details for next {months_ahead} months...")
    contracts = []
    for i in range(months_ahead):
        month = (datetime.today() + relativedelta(months=i)).strftime("%Y%m")
        contracts.append(vix_contract(month))

    app.expected_count = len(contracts)

    for contract in contracts:
        print(f"Requesting {contract.lastTradeDateOrContractMonth}")
        app.reqContractDetails(app.req_id, contract)
        app.req_id += 1

    app.done.wait(timeout=15)

    details = list(app.contract_details.values())
    print(f"\nFetched {len(details)} Contract Details:")
    for d in details:
        print(f" - {d['localSymbol']} | {d['lastTradeDateOrContractMonth']} | Exchange: {d['exchange']} | conId: {d['conId']}")
    return details


def request_fut_data(app, details, mode="historical"):
    print(f"\n=== Requesting Futures Data [{mode}] ===")

    app.reqMarketDataType(1)
    reqid_to_conid = {}
    all_rows = []

    for d in details:
        contract = vix_contract(d["lastTradeDateOrContractMonth"])
        reqId = app.req_id

        if mode == "live":
            print(f"Requesting snapshot for {d['localSymbol']}")
            app.reqMktData(reqId, contract, "", True, False, [])
        else:
            print(f"Requesting historical bar for {d['localSymbol']}")
            app.reqHistoricalData(reqId, contract, "", "1 D", "1 min", "MIDPOINT", 0, 1, False, [])

        reqid_to_conid[reqId] = d
        app.req_id += 1

        if mode == "historical":
            start = time.time()
            while not app.hist_data_ready.wait(timeout=1):
                elapsed = int(time.time() - start)
                print(f"Waiting for historical bar... {elapsed}s", end="\r")
            app.hist_data_ready.clear()

    if mode == "live":
        time.sleep(5)

    for reqId, data in app.data.items():
        desc = reqid_to_conid.get(reqId, {})
        row = {
            "ReqId": reqId,
            "LocalSymbol": desc.get("localSymbol", "N/A"),
            "Expiration": desc.get("lastTradeDateOrContractMonth", "N/A")
        }
        row.update(data)
        all_rows.append(row)

    if not all_rows:
        print("\nNo futures data received!")
        return

    df = pd.DataFrame(all_rows).set_index("ReqId")
    df = df[["LocalSymbol", "Expiration"] + [c for c in df.columns if c not in ["LocalSymbol", "Expiration"]]]

    print("\nFutures Data Summary:")
    print(tabulate(df, tablefmt=tabulate_formats[4], headers="keys"))

    today = datetime.today().strftime("%Y%m%d")
    suffix = "snapshot" if mode == "live" else "historical"
    filename = f"vix_futures_{suffix}_{today}.csv"
    df.to_csv(filename)
    print(f"\nSaved to {filename}")


def run_fetch(mode="historical"):
    print("\nStarting VIX futures data fetch...")
    app = BApi()
    app.connect("127.0.0.1", 4001, clientId=0)

    thread = threading.Thread(target=app.run, daemon=True)
    thread.start()
    time.sleep(1)

    details = get_contract_details(app, months_ahead=9)
    if len(details) < 2:
        print("Failed to fetch contract details.")
        app.disconnect()
        return

    request_fut_data(app, details, mode=mode)
    app.disconnect()
    print("\nProcess complete.\n")


if __name__ == "__main__":
    # Toggle here
    run_fetch(mode="historical")
    # run_fetch(mode="live")
