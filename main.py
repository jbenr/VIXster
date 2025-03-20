from just_do_it import *
from prep_data_2 import *


if __name__ == '__main__':
    df = load_data()
    data = feature_engineer(df)
    utils.pdf(data.tail(5))

    cluster_features = ['VIX', 'SP500_5d', 'SP500_drawdown', 'IG']
    cluster_regimes(data, n_clusters=7, cluster_features=cluster_features)

    feature_cols = [
        'VIX', '5-6', '7-8',
        'SP500_realized_vol_21d',
        'SP500_1d', 'SP500_5d',
        'SP500_drawdown'
    ]

    lin_model, preds = lin_regresh(data, feature_cols=feature_cols)
    utils.pdf(preds.head(30))
