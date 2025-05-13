import numpy as np
import pandas as pd
import itertools
import utils

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm

def compute_metrics(y_true, y_pred):
    valid_mask = (~pd.isna(y_true)) & (~pd.isna(y_pred))
    y_true_v = y_true[valid_mask]
    y_pred_v = y_pred[valid_mask]
    if len(y_true_v) == 0:
        return {"MAE": np.nan, "R2": np.nan, "IC": np.nan, "DirectionalAcc": np.nan}

    # MAE
    mae = np.mean(np.abs(y_true_v - y_pred_v))
    # R²
    r2 = r2_score(y_true_v, y_pred_v) if len(y_true_v) > 1 else np.nan
    # Information Coefficient (Pearson)
    if len(y_true_v) > 1:
        ic = np.corrcoef(y_true_v, y_pred_v)[0, 1]
    else:
        ic = np.nan
    # Directional Accuracy
    actual_signs = np.sign(y_true_v)
    pred_signs = np.sign(y_pred_v)
    directional_acc = np.mean(actual_signs == pred_signs)

    return {"MAE": mae, "R2": r2, "IC": ic, "DirectionalAcc": directional_acc}


def train_and_evaluate_cv(
    df,
    spread,
    active_blocks,
    month_cols,
    n_splits=5
):
    # Build the design matrix
    used_cols = []
    for b in active_blocks:
        if b == 'SEASONALITY_BLOCK':
            used_cols.extend(month_cols)
        else:
            used_cols.append(b)

    X_ = df[used_cols].copy()
    y_ = df[spread].copy()

    # Drop rows with missing data
    valid_idx = X_.dropna().index.intersection(y_.dropna().index)
    X_ = X_.loc[valid_idx]
    y_ = y_.loc[valid_idx]

    # TimeSeriesSplit from sklearn
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # We'll accumulate metrics across folds
    maes, r2s, ics, dirs = [], [], [], []
    final_model = None  # will store the last fold's model

    for train_idx, test_idx in tscv.split(X_):
        X_train, X_test = X_.iloc[train_idx], X_.iloc[test_idx]
        y_train, y_test = y_.iloc[train_idx], y_.iloc[test_idx]

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        fold_metrics = compute_metrics(y_test, y_pred)

        maes.append(fold_metrics["MAE"])
        r2s.append(fold_metrics["R2"])
        ics.append(fold_metrics["IC"])
        dirs.append(fold_metrics["DirectionalAcc"])

        final_model = model  # keep the last fold's model

    # average metrics across folds
    metrics = {
        "MAE": np.nanmean(maes),
        "R2": np.nanmean(r2s),
        "IC": np.nanmean(ics),
        "DirectionalAcc": np.nanmean(dirs)
    }
    return metrics, final_model


def all_subsets_feature_selection(
        df,
        target_cols,
        base_features,
        include_seasonality=True,
        parquet_path='all_subsets_results.parquet',
        n_splits=5,
        verbose=True
):
    data = df.copy()
    month_cols = [c for c in data.columns if c.startswith('Month_')]

    results = []
    best_models = {}

    # Build a final block list
    full_blocks = list(base_features)
    if include_seasonality and len(month_cols) > 0:
        full_blocks.append('SEASONALITY_BLOCK')

    for spread in target_cols:
        # For each spread, exclude that same spread from blocks
        candidate_blocks = []
        for b in full_blocks:
            if b != spread:
                candidate_blocks.append(b)

        # Generate all subsets
        subsets = []
        for r in range(1, len(candidate_blocks) + 1):
            for combo in itertools.combinations(candidate_blocks, r):
                subsets.append(list(combo))

        if verbose:
            print(f"\nSpread '{spread}': {len(subsets)} subsets to evaluate with {n_splits}-fold CV")

        best_mae = np.inf
        best_subset = None
        best_metrics = None
        best_model = None

        for subset_blocks in tqdm(subsets, desc=f"Spread {spread}", leave=False):
            metrics, model = train_and_evaluate_cv(
                df=data,
                spread=spread,
                active_blocks=subset_blocks,
                month_cols=month_cols,
                n_splits=n_splits
            )

            row = {
                'Spread': spread,
                'SubsetBlocks': subset_blocks,
                'MAE': metrics['MAE'],
                'R2': metrics['R2'],
                'IC': metrics['IC'],
                'DirectionalAcc': metrics['DirectionalAcc']
            }
            results.append(row)

            # pick best by MAE
            if metrics['MAE'] < best_mae:
                best_mae = metrics['MAE']
                best_subset = subset_blocks
                best_metrics = metrics
                best_model = model

        if verbose:
            print(f"Best subset for spread '{spread}' => {best_subset}, MAE={best_mae:.4f}")

        best_models[spread] = {
            'best_subset': best_subset,
            'metrics': best_metrics,
            'model': best_model
        }

    # Convert to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_parquet(parquet_path, index=False)
    if verbose:
        print(f"\nAll subset CV results saved to {parquet_path}")

    return best_models

from prep_data_2 import load_data, feature_engineer

if __name__ == "__main__":
    df = load_data()
    data = feature_engineer(df)
    utils.pdf(data.tail(5))

    feature_cols = [
        'VIX', 'DFR',
        'SP500_realized_vol_7d', 'SP500_realized_vol_14d',
        'SP500_realized_vol_21d', 'SP500_realized_vol_50d',
        'SP500_1d', 'SP500_5d',
        'SP500_drawdown'
    ] + [col for col in data.columns if col.startswith("Month_")]

    base_feats = [
        'VIX', 'DFR',
        'SP500_realized_vol_7d',
        'SP500_realized_vol_21d', 'SP500_realized_vol_50d',
        'SP500_1d', 'SP500_5d',
        'SP500_drawdown',
        '1-2','2-3','3-4','4-5','5-6','6-7','7-8'  # i.e. you can put other spreads here
    ]

    # The target spreads we want to model individually
    target_spreads = [f"{i}-{i+1}" for i in range(1,8)]

    # best_results = all_subsets_feature_selection(
    #     df=data,
    #     target_cols=target_spreads,
    #     base_features=base_feats,
    #     include_seasonality=True,
    #     parquet_path='data/cv_all_subsets.parquet',
    #     n_splits=5,
    #     verbose=True
    # )
    #
    # # Inspect final best model per spread
    # for spread, info in best_results.items():
    #     print(f"\nSpread: {spread}")
    #     print("Best Subset:", info['best_subset'])
    #     print("Avg CV Metrics:", info['metrics'])
    #     # info['model'] is the last fold's model from the CV

    cv = pd.read_parquet('data/cv_all_subsets.parquet')
    opt = []
    utils.pdf(cv[:10])
    for spread in list(cv['Spread'].unique()):
        temp = cv[cv['Spread'] == spread]
        print(f"== {spread} ==")
        mae = temp.loc[temp['MAE'].idxmin()]
        r2 = temp.loc[temp['R2'].idxmax()]
        ic = temp.loc[temp['IC'].idxmax()]
        da = temp.loc[temp['DirectionalAcc'].idxmax()]
        opt.append(mae)
        opt.append(r2)
        opt.append(ic)
        opt.append(da)
        print(f"MAE {list(mae['SubsetBlocks'])}")
        print(f"R2 {list(r2['SubsetBlocks'])}")
        print(f"IC {list(ic['SubsetBlocks'])}")
        print(f"DA {list(da['SubsetBlocks'])}")
    opt = pd.DataFrame(opt, columns=cv.columns).reset_index(drop=True)
    print(opt.tail(5))
    opt.to_parquet('data/optimal_models.parquet')
