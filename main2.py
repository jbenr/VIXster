# main2.py

import utils
import pandas as pd
from prep_torch import TradingDataset
from modelo2 import MultiHeadTradingModel, train_model, predict
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
import torch


if __name__ == '__main__':
    file_path = "data/features/final_df.parquet"
    df = pd.read_parquet(file_path)
    df["Trade_Date"] = pd.to_datetime(df["Trade_Date"]).dt.date
    utils.pdf(df.tail(3))

    # Identify market-wide (global) features
    excluded_cols = ["Trade_Date", "Spread_ID", "Feature_Name", "Feature_Value", "Spread_Value", "Spread_Index"]
    global_feature_cols = [col for col in df.columns if col not in excluded_cols]

    # Encode categorical columns
    categorical_cols = ["day_of_week", "day_of_month", "month", "Macro_Regime"]
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Sort data for proper grouping
    df = df.sort_values(by=["Trade_Date", "Spread_ID"]).reset_index(drop=True)

    # Extract latest trade date
    latest_trade_date = df["Trade_Date"].max()
    df_latest = df[df["Trade_Date"] == latest_trade_date]

    print(f"Latest Trade Date: {latest_trade_date}")

    # Define dataset and dataloader
    dataset = TradingDataset(df, global_feature_cols)
    batch_size = 256  # 🚀 Try increasing batch size (if memory allows)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # 🚀 Keep False for time-series
        # num_workers=0,  # ✅ Good for CPU parallelism
        pin_memory=True,  # 🚀 Faster CPU-to-GPU transfers
        # prefetch_factor=16,  # 🔥 Increase prefetching for better GPU utilization
        # persistent_workers=True,  # ✅ Prevent worker resets
        drop_last=True  # 🔥 Prevents partial batches slowing down training
    )

    # moving dataloader to GPU 
    for i, batch in enumerate(dataloader):
        batch = [x.to("cuda", non_blocking=True) for x in batch]
        break

    # Check batch output
    for global_features, spread_features, spread_masks, target, spread_indices in dataloader:
        print(f"Global Features shape: {global_features.shape}")  # (batch_size, num_global_features)
        print(f"Spread Features shape: {spread_features.shape}")  # (batch_size, num_spreads, max_features_per_spread)
        print(f"Spread Masks shape: {spread_masks.shape}")
        print(f"Target shape: {target.shape}")  # (batch_size, num_spreads)
        print(f"Spread indices shape: {spread_indices.shape}")  # (batch_size, num_spreads)
        break  # Show only first batch

    num_global_features = len(global_feature_cols)  # ✅ Number of macro/global features
    num_spreads = df["Spread_ID"].nunique()  # ✅ Number of unique spreads
    num_spread_features = df["Feature_Name"].nunique()  # ✅ Number of unique spread-specific features
    print(num_global_features, num_spreads, num_spread_features)

    num_global_features = 21  # From your dataset
    num_spreads = 7  # Number of unique spreads
    num_spread_features = 24  # Each spread has 24 features
    print(num_global_features, num_spreads, num_spread_features)

    # Instantiate model
    model = MultiHeadTradingModel(num_global_features, num_spreads, num_spread_features)
    # model = torch.compile(model)  # 🚀 Optimize model execution

    print("Starting training...")
    train_model(model, dataloader, num_epochs=10, learning_rate=0.001)

    # Predict with trained model
    predictions = predict(model, dataloader)

    # Print shape of predictions (should be [num_samples, num_spreads, 24])
    print("Predictions shape:", predictions.shape)

    # Extract the last batch of predictions (latest day)
    latest_predictions = predictions[-1]  # Last batch corresponds to latest day

    # Extract actual target values for the latest trade date
    actual_values = df_latest.groupby("Spread_ID")["Spread_Value"].first().values

    # Print predictions vs actuals for the latest trade date
    print("\nLatest Day Predictions vs Actual Values:")
    for spread_idx, (pred, actual) in enumerate(zip(latest_predictions, actual_values)):
        print(f"Spread {spread_idx + 1}: Predicted={pred.item():.4f}, Actual={actual:.4f}")