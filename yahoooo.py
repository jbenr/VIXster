import random
import time
import atexit
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from requests_ip_rotator import ApiGateway, EXTRA_REGIONS
import utils
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By


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


def scraper(url: str, css_selector: str) -> float:
    """
    Spins up a headless Chrome driver, opens `url`, finds the element
    matching `css_selector`, and returns its text parsed as float.
    Raises an exception if anything goes wrong.
    """
    # Path to your chromedriver binary; adjust if it’s elsewhere
    service = Service("/usr/local/bin/chromedriver")
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")

    driver = webdriver.Chrome(service=service, options=options)
    try:
        driver.get(url)
        # tiny sleep in case the page needs a moment to render
        time.sleep(1)

        elem = driver.find_element(By.CSS_SELECTOR, css_selector)
        text = elem.text.strip().replace(",", "")  # remove commas if present
        text = text.strip().replace("$", "")  # remove $ if present

        price = float(text)
    finally:
        driver.quit()

    return price


def get_price(symbol: str) -> pd.DataFrame | None:
    """
    Fetch the latest close for a single ticker or index name.
    If the user passes "VIX" or "^VIX", we attempt to scrape CBOE first.
    Otherwise, or if scraping fails, we fall back to yfinance.
    Returns a 1×1 DataFrame (index=today’s date, column="Close"), or None on failure.
    """

    # 1) Normalize the incoming symbol into a “base” name:
    if symbol == "VIX" or symbol == "^VIX":
        base_symbol = "VIX"
    elif symbol == "SP500" or symbol == "^GSPC":
        base_symbol = "SP500"
    else:
        return None  # e.g. "VVIX", "BTC", "AAPL", etc.

    # 2) Define scraping rules keyed by the base name:
    scraping_info = {
        "VIX": {
            "url": "https://www.cboe.com/tradable_products/vix/",
            "css": "h2.tw-text-ink-brand-secondary.tw-font-subheading-lg",
            # If scraping fails, we’ll fallback to yfinance("^VIX").
        },
        "SP500": {
            "url": "https://www.cnbc.com/quotes/.SPX",
            "css": (
                "#quote-page-strip > div.QuoteStrip-dataContainer "
                "> div.QuoteStrip-lastTimeAndPriceContainer "
                "> div.QuoteStrip-lastPriceStripContainer "
                "> span.QuoteStrip-lastPrice"
            ),
            # If scraping fails, we’ll fallback to yfinance("^GSPC").
        },
    }

    # 3) If base_symbol is exactly "VIX" or "SP500", attempt to scrape first:
    if base_symbol in scraping_info:
        info = scraping_info[base_symbol]
        try:
            # avoid hammering the site
            time.sleep(random.uniform(0.5, 1.5))
            live_price = scraper(info["url"], info["css"])
            today = datetime.now().date()
            return pd.DataFrame({"Close": [live_price]}, index=[today])

        except Exception as e:
            print(f"Scraping for {base_symbol} failed: {e}. Falling back to yfinance…")

    # 4) FALLBACK: use yfinance.  But if base_symbol was "VIX"/"SP500",
    #    we need to map back to the proper “caret” ticker:
    yf_symbol = symbol
    if base_symbol == "VIX":
        yf_symbol = "^VIX"
    elif base_symbol == "SP500":
        yf_symbol = "^GSPC"
    # Otherwise (e.g. base_symbol = "VVIX" or "BTC"), we just use the original `symbol`.

    try:
        time.sleep(random.uniform(0.5, 1.5))
        t = yf.Ticker(yf_symbol)
        df = t.history(period="1d", timeout=10)
        df.index = pd.to_datetime(df.index).date
        return df[["Close"]].tail(1)

    except Exception as e:
        print(f"Error retrieving {yf_symbol} via yfinance: {e}")
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
    # None
