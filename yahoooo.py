import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import utils


def fetch_yahoo_finance_data(symbols, start_date, end_date):
    """
    Fetch historical daily stock data for given symbols from Yahoo Finance.
    """
    data = yf.download(symbols, start=start_date, end=end_date, group_by='ticker')
    data = filter_adjusted_close(data)
    return data

def filter_adjusted_close(data):
    """
    Filter the DataFrame to only include the adjusted close columns for each ticker.
    """
    adj_close_columns = [(symbol, 'Adj Close') for symbol in data.columns.levels[0]]
    filtered_data = data.loc[:, adj_close_columns]
    # Flatten the multi-index columns
    filtered_data.columns = [symbol for symbol, _ in filtered_data.columns]
    return filtered_data

def get_price(x):
    try:
        t = yf.Ticker(x)
        data = t.history(period="1d")
        data.index = pd.to_datetime(data.index).date
        latest_price = data[['Close']].tail(1)
        return latest_price
    except Exception as e:
        print(f"Error retrieving {x} price: {e}")
        return None

def run():
    # List of stock symbols to fetch data for
    stonks = [
        '^VIX', '^GSPC', '^VVIX',
        '^IRX','^FVX','^TNX','^TYX',
        'GC=F','HG=F','CL=F'
    ]
    tickers = stonks

    # Calculate the start and end date
    end_date = datetime.now()
    start_date = end_date - timedelta(days=14 * 365)  # 15 years back

    print(f"Fetching data for {tickers} from {start_date.date()} to {end_date.date()}...")
    data = fetch_yahoo_finance_data(tickers, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    data.index = pd.to_datetime(data.index).date

    if isinstance(data, pd.DataFrame):
        utils.pdf(data.tail(10))
        data.to_parquet('data/yahoo.parquet')
    else:
        print("Failed to fetch data")

def main():
    v = get_price("^VIX")
    print(v)

    s = get_price("^GSPC")
    print(s)

if __name__ == "__main__":
    main()
