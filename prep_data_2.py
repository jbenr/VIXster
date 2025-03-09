import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import vixy
import utils
import talib

def prep_X_y(params):
    y = vixy.process_vix()
    y = vixy.pivot_vix(y)
    y.index = pd.to_datetime(y.index).date

    X = y[['DTR', 'DFR']]
    y = vixy.vix_spreads(y)
    y = y[['1-2', '2-3', '3-4', '4-5', '5-6', '6-7', '7-8']]

    fred = pd.read_parquet('data/fred.parquet')
    fred.index = pd.to_datetime(fred.index).date
    fred.index.name = 'Trade_Date'

    stonk = fred[['VIX', 'SP500', '10Y']].copy()
    for col in ['VIX', '10Y']: stonk[f'{col}_chg'] = fred[col].diff()
    for col in ['SP500']: stonk[f'{col}_pct_chg'] = fred[col].pct_change(fill_method=None)

    shared_indexes = X.index.intersection(stonk.index)
    stonk = stonk.loc[shared_indexes]

    X = pd.merge(X, stonk, how='left', left_index=True, right_index=True).dropna()
    shared_indexes = X.index.intersection(y.index)

    X = X.loc[shared_indexes]
    y = y.loc[shared_indexes]

    bb_period = params.get("BB_period", 14)
    rsi_period = params.get("RSI_period", 14)

    dic_dat = {} # generate technical indicators for each spread
    for col in y.columns:
        tech = y[[col]].copy()

        tech[f'Upper_BB_{bb_period}'], tech[f'Middle_BB_{bb_period}'], tech[f'Lower_BB_{bb_period}'] = talib.BBANDS(
            tech[col], timeperiod=bb_period, nbdevup=2, nbdevdn=2, matype=0)
        tech[f'RSI_{rsi_period}'] = talib.RSI(tech[col], rsi_period)-50

        tech.drop(columns=col,inplace=True)
        dic_dat[col] = tech.dropna()

    shared_indexes = dic_dat[y.columns[0]].index.intersection(X.index)
    X = X.loc[shared_indexes]
    y = y.loc[shared_indexes]

    return X, y, dic_dat


X, y, dat = prep_X_y({'BB_period': 14, 'RSI_period': 14})
utils.pdf(X.tail(10))
utils.pdf(y.tail(10))
print(dat.keys())
for key in dat.keys():
    print(key)
    utils.pdf(dat[key].tail(10))


def feat_imp(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse:.4f}")

    plot_importance(model)
    plt.show()

    importance_scores = model.feature_importances_
    for i, v in enumerate(importance_scores):
        print(f'Feature: {X.columns[i]}, Score: {v}')

# for col in y.columns:
#     print(col)
#     feat_imp(X, y[col])
#     print('\n')


from statsmodels.stats.outliers_influence import variance_inflation_factor
def calculate_vif(X):
    # Ensure no NaN or Inf values in X
    X_clean = X.replace([np.inf, -np.inf], np.nan)
    X_clean = X_clean.fillna(X_clean.mean())

    vif_data = pd.DataFrame()
    vif_data["Feature"] = X_clean.columns
    vif_data["VIF"] = [variance_inflation_factor(X_clean.values, i) for i in range(X_clean.shape[1])]

    return vif_data

vif_df = calculate_vif(X)
print(vif_df)


