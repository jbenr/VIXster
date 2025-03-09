import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd


# ===== Transformer Model for Multi-Output Spread Forecasting =====

class VIXSpreadTransformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, dropout=0.1):
        super(VIXSpreadTransformer, self).__init__()

        # Linear Projection for Inputs
        self.input_projection = nn.Linear(input_dim, model_dim)

        # Transformer Encoder
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, dropout=dropout),
            num_layers=num_layers
        )

        # Output Linear Layer (7 Spreads)
        self.output_layer = nn.Linear(model_dim, 7)

    def forward(self, x):
        # Pass input through linear embedding
        x = self.input_projection(x)

        # Apply Transformer Encoder
        x = self.transformer(x)

        # Output 7 Spreads per timestep
        return self.output_layer(x)


# ===== Data Preparation =====

class VIXDataset(Dataset):
    def __init__(self, features, targets, seq_length=30):
        self.features = features
        self.targets = targets
        self.seq_length = seq_length

    def __len__(self):
        return len(self.features) - self.seq_length

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_length]
        y = self.targets[idx + self.seq_length]  # Predict next timestep
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# ===== Training Function =====

def train_transformer(model, dataloader, epochs=10, lr=0.001):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()

    for epoch in range(epochs):
        epoch_loss = 0
        for x_batch, y_batch in dataloader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss / len(dataloader)}")


# ===== Running the Model =====

def main():
    # Load processed feature dataset
    data = pd.read_parquet("data/processed_vix_spreads.parquet")

    # Select relevant features
    feature_cols = [col for col in data.columns if "zscore" in col or "corr" in col or "DTR" in col or "DFR" in col]
    target_cols = ["1-2", "2-3", "3-4", "4-5", "5-6", "6-7", "7-8"]

    # Convert to numpy
    X = data[feature_cols].values
    y = data[target_cols].values

    # Normalize (optional)
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Create dataset & dataloader
    dataset = VIXDataset(X, y, seq_length=30)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    # Initialize Model
    model = VIXSpreadTransformer(input_dim=len(feature_cols), model_dim=64, num_heads=4, num_layers=4)

    # Train Model
    train_transformer(model, dataloader, epochs=10)

    # Save Model
    torch.save(model.state_dict(), "vix_spread_transformer.pth")
    print("Model training complete & saved.")


if __name__ == "__main__":
    main()
