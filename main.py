# main.py

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

    # Define dataset and dataloader
    dataset = TradingDataset(df, global_feature_cols)
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    num_global_features = 21  # From your dataset
    num_spreads = 7  # Number of unique spreads
    num_spread_features = 24  # Each spread has 24 features

    # Instantiate model
    model = MultiHeadTradingModel(num_global_features, num_spreads, num_spread_features)

    print("Starting training...")
    train_model(model, dataloader, num_epochs=5, learning_rate=0.001)

    # Predict with trained model
    predictions = predict(model, dataloader)

    # Print shape of predictions (should be [num_samples, num_spreads, 24])
    print("Predictions shape:", predictions.shape)
