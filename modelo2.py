import torch
import torch.nn as nn
import torch.optim as optim


class MultiHeadTradingModel(nn.Module):
    def __init__(self, num_global_features, num_spreads, num_spread_features):
        super(MultiHeadTradingModel, self).__init__()

        self.num_spreads = num_spreads  # Number of spread heads

        # Shared input layer (processes spread features)
        self.shared_fc = nn.Linear(num_spread_features, 32)  # Small hidden layer

        # Separate heads for each spread (each outputs 24 targets)
        self.spread_heads = nn.ModuleList([
            nn.Linear(32, 24) for _ in range(num_spreads)  # Each head maps to 24 outputs
        ])

    def forward(self, spread_features):
        """
        spread_features: [batch_size, num_spreads, num_spread_features]
        """
        batch_size = spread_features.shape[0]

        # Process each spread independently
        outputs = []
        for i in range(self.num_spreads):
            spread_input = spread_features[:, i, :]  # Select spread i features [batch_size, num_spread_features]
            hidden = torch.relu(self.shared_fc(spread_input))  # Shared transformation
            spread_output = self.spread_heads[i](hidden)  # Spread-specific output [batch_size, 24]
            outputs.append(spread_output)

        # Stack outputs: [batch_size, num_spreads, 24]
        return torch.stack(outputs, dim=1)


from torch.cuda.amp import autocast, GradScaler
import time

def train_model(model, dataloader, num_epochs=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scaler = GradScaler()  # Mixed precision training

    print(f"Number of batches: {len(dataloader)}")
    if len(dataloader) == 0:
        raise ValueError("Dataloader is empty! Check dataset loading.")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for batch_idx, (global_features, spread_features, targets, spread_indices) in enumerate(dataloader):
            spread_features, targets = spread_features.to(device), targets.to(device)

            optimizer.zero_grad()
            with autocast():  # Mixed precision for speed
                predictions = model(spread_features)
                loss = criterion(predictions, targets)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            # Print every 10 batches instead of every batch
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}, Time: {time.time() - start_time:.2f}s")

    print("Training complete!")


# Define prediction function
def predict(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()  # Set to evaluation mode

    all_predictions = []
    with torch.no_grad():  # No need to track gradients
        for global_features, spread_features, targets, spread_indices in dataloader:
            spread_features = spread_features.to(device)
            predictions = model(spread_features)  # Forward pass

            all_predictions.append(predictions.cpu())  # Move predictions to CPU

    return torch.cat(all_predictions, dim=0)  # Concatenate predictions across batches

