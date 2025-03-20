import pandas as pd

from just_do_it import *
from just_do_it import perf_eval
from prep_data_2 import *


if __name__ == '__main__':
    df = load_data()
    data = feature_engineer(df)
    # utils.pdf(data.tail(5))

    cluster_features = ['VIX', 'SP500_5d', 'SP500_drawdown', 'IG']
    clust = cluster_regimes(data, n_clusters=8, cluster_features=cluster_features)

    feature_cols = [
        'VIX', '5YBEI',
        'SP500_realized_vol_21d',
        'SP500_1d', 'SP500_5d',
        'SP500_drawdown'
    ] + [col for col in data.columns if col.startswith("Month_")]

    lin_model, preds = lin_regresh(data, feature_cols=feature_cols)

    # evaluating performance by cluster!
    for spread in list(preds.keys()):
        print(f"\n== Spread {spread} ==")
        spr = preds[spread]
        clusts = clust['ClusterRegime'].unique()
        clusts.sort()
        for cluster in clusts:
            temp = spr.loc[spr.index.isin(clust[clust['ClusterRegime'] == cluster].index)]
            try:
                metrics = perf_eval(temp['Actual'], temp['Pred'])
                print(f"Cluster {cluster} [{len(temp)} samples] - MAE: {metrics['MAE']:.5f} | R²: {metrics['R²']:.5f} | IC: {metrics['IC']:.5f} | Directional Acc: {metrics['Directional Acc']:.2%}")
            except Exception as e: print(e)

    print('\n== Preds ==')
    utils.pdf(preds[list(preds.keys())[0]].head(3))
