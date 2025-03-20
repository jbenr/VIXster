from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
import pandas as pd
import numpy as np
import utils


def lin_regresh(df, feature_cols=['VIX', '1-2', '2-3']):
    data = df.copy()

    target_cols = [f'{i}-{i+1}' for i in range(1, 8)]  # 1-2, 2-3,... 7-8
    # target_cols = ['6-7']  # ✅ Uncomment this to test a single spread

    print(f"\n== Feature cols ==\n{feature_cols}\n")
    X = data[feature_cols]
    Y = data[target_cols]

    # Align to predict next-day spreads: use X_t to predict Y_{t+1}
    Y_next = Y.shift(-1)  # Shift targets up by 1
    X = X.iloc[:-1]       # Drop last row to match target size
    Y_next = Y_next.iloc[:-1]

    X = X.dropna()
    Y_next = Y_next.loc[X.index]  # Keep corresponding Y values

    split = int(0.7 * len(X))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    Y_train, Y_test = Y_next.iloc[:split], Y_next.iloc[split:]

    model = LinearRegression()
    model.fit(X_train, Y_train)

    Y_pred = model.predict(X_test)

    # Create dictionary to store individual results per spread
    spread_results = {}

    for i, spread in enumerate(target_cols):
        actuals = Y_test.iloc[:, i]
        preds = Y_pred[:, i]
        resids = actuals - preds

        # Store results in dictionary
        spread_results[spread] = pd.DataFrame({
            "Actual": actuals.values,
            "Pred": preds,
            "Resid": resids
        })

        metrics = perf_eval(actuals, preds)
        print(f"Spread {spread} - MAE: {metrics['MAE']:.5f} | R²: {metrics['R²']:.5f} | IC: {metrics['IC']:.5f} | Directional Acc: {metrics['Directional Acc']:.2%}")

    coef_df = pd.DataFrame(model.coef_, columns=feature_cols, index=target_cols)
    print('== Coefficient importance ==')
    utils.pdf(coef_df.round(3))
    return model, spread_results


def perf_eval(actuals, preds):
    mae = mean_absolute_error(actuals, preds)
    r2 = r2_score(actuals, preds)
    ic = np.corrcoef(actuals, preds)[0, 1]

    actual_signs = np.sign(actuals)
    pred_signs = np.sign(preds)
    directional_acc = np.mean(actual_signs == pred_signs)

    return {
        "MAE": mae,
        "R²": r2,
        "IC": ic,
        "Directional Acc": directional_acc
    }


if __name__ == "__main__":
    print('.')