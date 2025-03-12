import pandas as pd
import utils
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


class TradingDataset(Dataset):
    def __init__(self, df, global_feature_cols):
        self.df = df.copy()
        self.trade_dates = sorted(self.df["Trade_Date"].unique())  # Unique trade days
        self.spread_ids = sorted(self.df["Spread_ID"].unique())  # Unique spreads
        self.global_feature_cols = global_feature_cols  # Market-wide features

        self.grouped_data = self._process_data()

    def _process_data(self):
        """ Convert long-format spread-specific features + global features into structured tensors. """
        grouped_data = {}

        for trade_date in tqdm(self.trade_dates, desc="Processing Trade Dates", unit="day"):
            df_day = self.df[self.df["Trade_Date"] == trade_date]

            global_features = torch.tensor(df_day[self.global_feature_cols].iloc[0].values, dtype=torch.float32)

            spread_data = {}
            max_feature_count, max_targets_count = 0, 0

            for spread in self.spread_ids:
                df_spread = df_day[df_day["Spread_ID"] == spread]

                if df_spread.empty: continue  # Skip if no data for this spread on this day

                feature_matrix = df_spread.pivot(index="Spread_ID", columns="Feature_Name", values="Feature_Value")
                spread_features = torch.tensor(feature_matrix.fillna(0).values, dtype=torch.float32)

                if spread_features.dim() == 1:
                    spread_features = spread_features.unsqueeze(0)  # Ensure it's 2D (1 row, features)

                spread_data[spread] = spread_features
                max_feature_count = max(max_feature_count, spread_features.shape[-1])  # Keep track of max feature count

                # Count targets
                num_targets = len(df_spread["Spread_Value"].values)
                max_targets_count = max(max_targets_count, num_targets)

            # Store processed data for the date
            grouped_data[trade_date] = {
                "global_features": global_features,
                "spread_features": spread_data,
                "max_features": max_feature_count,
                "max_targets": max_targets_count
            }

        return grouped_data

    def __len__(self):
        return len(self.trade_dates)

    def __getitem__(self, idx):
        """ Return (global features, spread-specific features, targets, spread indices) """
        trade_date = self.trade_dates[idx]
        data = self.grouped_data[trade_date]

        global_features = data["global_features"]
        spread_data = data["spread_features"]
        max_features = data["max_features"]
        max_targets = data["max_targets"]

        features_list, targets_list, spread_indices_list = [], [], []

        for spread_idx, spread in enumerate(self.spread_ids):
            if spread in spread_data:
                spread_features = spread_data[spread]

                # Pad features to max feature size
                pad_size = max_features - spread_features.shape[-1]
                if pad_size > 0:
                    pad_tensor = torch.zeros((spread_features.shape[0], pad_size), dtype=torch.float32)
                    spread_features = torch.cat([spread_features, pad_tensor], dim=1)

                features_list.append(spread_features.squeeze(0))  # 🔥 Remove unnecessary dimension

                # Extract target
                target_values = self.df[(self.df["Trade_Date"] == trade_date) & (self.df["Spread_ID"] == spread)]["Spread_Value"].values
                target_tensor = torch.tensor(target_values, dtype=torch.float32)

                # Pad target to max_targets size
                target_pad_size = max_targets - target_tensor.shape[0]
                if target_pad_size > 0:
                    pad_tensor = torch.zeros(target_pad_size, dtype=torch.float32)
                    target_tensor = torch.cat([target_tensor, pad_tensor])

                targets_list.append(target_tensor)
                spread_indices_list.append(spread_idx)  # 🔥 Directly append spread index (removing singleton dim)

        # 🔥 Fix stacking dimensions (Remove extra dimension)
        spread_features = torch.stack(features_list, dim=0) if features_list else torch.empty(0)  # 🔥 Remove singleton dim
        targets = torch.stack(targets_list, dim=0) if targets_list else torch.empty(0)
        spread_indices = torch.tensor(spread_indices_list, dtype=torch.long)  # 🔥 Convert list to tensor

        return global_features, spread_features, targets, spread_indices






if __name__ == "__main__":

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
    batch_size = 16
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Check batch output
    for global_features, spread_features, target, spread_indices in dataloader:
        print(f"Global Features shape: {global_features.shape}")  # (batch_size, num_global_features)
        print(f"Spread Features shape: {spread_features.shape}")  # (batch_size, num_spreads, max_features_per_spread)
        print(f"Target shape: {target.shape}")  # (batch_size, num_spreads)
        print(f"Spread indices shape: {spread_indices.shape}")  # (batch_size, num_spreads)
        break  # Show only first batch
