# main.py

import utils
import pandas as pd
from prep_data import prep_X_y

def main():
    # Prepare the data using your existing prep_data functions
    X, y, dic_dat = prep_X_y({'BB_period': 14, 'RSI_period': 14})
    print("Latest X data:")
    utils.pdf(X.tail(10))
    print("Latest y data:")
    utils.pdf(y.tail(10))

    fred = pd.read_parquet('data/fred.parquet')
    utils.pdf(fred.tail(10))


if __name__ == '__main__':
    main()
