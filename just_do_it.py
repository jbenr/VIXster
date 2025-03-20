from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

def lin_regresh(df, feature_cols=['VIX', '1-2', '2-3']):
    data = df.copy()

    # target_cols = [f'{i}-{i+1}' for i in range(1, 8)]  # 1-2, 2-3,... 7-8
    target_cols = ['6-7']

    X = data[feature_cols]
    Y = data[target_cols]

    # Align to predict next-day spreads: use X_t to predict Y_{t+1}
    Y_next = Y.shift(-1)  # Shift targets up by 1
    X = X.iloc[:-1]       # Drop last row to match target size
    Y_next = Y_next.iloc[:-1]

    # Handle potential NaNs
    X = X.dropna()
    Y_next = Y_next.loc[X.index]  # Keep corresponding Y values

    # Split into training and test sets (e.g., 70% train, 30% test)
    split = int(0.7 * len(X))
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    Y_train, Y_test = Y_next.iloc[:split], Y_next.iloc[split:]

    # Train linear regression on training data
    model = LinearRegression()
    model.fit(X_train, Y_train)

    # Make predictions on test set
    Y_pred = model.predict(X_test)

    # Evaluate Performance
    mse = mean_squared_error(Y_test, Y_pred)
    r2 = r2_score(Y_test, Y_pred)

    print(f"🔹 Mean Squared Error (MSE): {mse:.5f}")
    print(f"🔹 R² Score: {r2:.5f}")

    # Create a DataFrame for actual vs predicted values
    results = pd.DataFrame({
        "Actual": Y_test.mean(axis=1),  # Mean across all spreads
        "Predicted": Y_pred.mean(axis=1)  # Mean prediction
    })
    results['Resid'] = results['Actual'] - results['Predicted']

    return model, results


import numpy as np
from sklearn.metrics import mean_squared_error

def perf_eval(data, X_test, Y_test, Y, Y_pred):
    # Calculate MSE for each spread
    mse_values = []
    for i in range(Y_test.shape[1]):  # 7 spreads
        mse = mean_squared_error(Y_test.iloc[:, i], Y_pred[:, i])
        mse_values.append(mse)
    mse_avg = np.mean(mse_values)
    print(f"MSE Avg: {mse_avg}")

    # Calculate Information Coefficient (Pearson correlation) for each spread
    ic_values = []
    for i in range(Y_test.shape[1]):
        corr = np.corrcoef(Y_test.iloc[:, i], Y_pred[:, i])[0,1]
        ic_values.append(corr)
    ic_avg = np.mean(ic_values)
    print(f"IC Avg: {ic_avg}")

    # Calculate directional accuracy for each spread
    direction_acc = []
    # We compare the sign of (predicted change) vs (actual change) for each day and spread
    Y_actual_today = Y.loc[X_test.index]  # actual spreads on day t (the baseline for change)
    for i, idx in enumerate(X_test.index):
        actual_today = Y_actual_today.loc[idx].values   # S_t actual
        actual_next = Y_test.loc[idx].values            # S_{t+1} actual
        pred_next = Y_pred[i]                           # S_{t+1} predicted
        # mark correct if predicted and actual changes have same sign
        direction_corr = ((actual_next - actual_today) * (pred_next - actual_today) > 0)
        direction_acc.extend(direction_corr.astype(int))  # True/False -> 1/0
    directional_accuracy = np.mean(direction_acc)
    print(f"Directional Accuracy: {directional_accuracy}")
    # perf_eval_regime(data, X_test, Y_test, Y_pred, direction_acc)


def perf_eval_regime(data, X_test, Y_test, Y_pred, direction_acc):
    # Identify test-set days by regime
    test_regimes = data.loc[X_test.index, 'ClusterRegime']
    mse_calm = mean_squared_error(Y_test[test_regimes==1], Y_pred[test_regimes==1])
    mse_vol  = mean_squared_error(Y_test[test_regimes==0], Y_pred[test_regimes==0])
    dir_acc_calm = np.mean([d for i,d in enumerate(direction_acc) if test_regimes.iloc[i]==1])
    dir_acc_vol  = np.mean([d for i,d in enumerate(direction_acc) if test_regimes.iloc[i]==0])
    print(f"MSE (Calm) = {mse_calm:.3f}, MSE (Volatile) = {mse_vol:.3f}")
    print(f"Directional Accuracy (Calm) = {dir_acc_calm*100:.1f}%, (Volatile) = {dir_acc_vol*100:.1f}%")


if __name__ == "__main__":
    print('.')