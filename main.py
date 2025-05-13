import pandas as pd
import pickle

from just_do_it import *
from just_do_it import perf_eval
from prep_data_2 import *
from single_regresh import train_and_evaluate_cv


if __name__ == '__main__':
    df = load_data()
    data = feature_engineer(df)
    utils.pdf(data.tail(5))

    cluster_features = [
        'VIX',
        'SP500_realized_vol_21d',
        'SP500_1d', 'SP500_5d',
        'SP500_drawdown'
    ]
    clust = cluster_regimes(data, n_clusters=5, cluster_features=cluster_features)

    # The target spreads we want to model individually
    target_spreads = [f"{i}-{i+1}" for i in range(1,8)]
    utils.make_dir('data/models')
    best_feats = pd.read_parquet('data/optimal_models.parquet')
    for spread in list(best_feats['Spread'].unique()):
        cols = best_feats[best_feats['Spread']==spread].loc[
            best_feats[best_feats['Spread']==spread]['DirectionalAcc'].idxmax()
        ]['SubsetBlocks']
        print(f"{spread} {list(cols)}")

        metrics, final_model = train_and_evaluate_cv(
            df=data, spread=spread, active_blocks=cols,
            month_cols=[c for c in data.columns if c.startswith('Month_')],
            n_splits=5
        )
        with open(f"data/models/linear_model_{spread}.pkl", "wb") as f:
            pickle.dump(final_model, f)
        print(f"Saved {spread} linear model.")

