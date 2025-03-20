import torch
import torch.nn as nn
import torch.optim as optim

# Torch optimizations
import torch.backends.cudnn as cudnn
import torch._inductor

cudnn.benchmark = True
torch._inductor.config.triton.cudagraphs = True
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class MultiHeadTradingModel(nn.Module):
    def __init__(self, num_global_features, num_spreads, num_spread_features):
        super(MultiHeadTradingModel, self).__init__()

        self.num_spreads = num_spreads

        # Shared feature extraction for spread-specific features
        self.spread_fc = nn.Sequential(
            nn.Linear(num_spread_features, 64),
            nn.ReLU(),
            nn.LayerNorm(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.LayerNorm(32)
        )

        # New: Global feature processing
        self.global_fc = nn.Sequential(
            nn.Linear(num_global_features, 16),  # ✅ Reduce global features to meaningful size
            nn.ReLU(),
            nn.LayerNorm(16)
        )

        # Spread heads: now take combined macro + spread data
        self.spread_heads = nn.ModuleList([
            nn.Linear(32 + 16, 1) for _ in range(num_spreads)  # ✅ Now includes macro influence
        ])

    def forward(self, global_features, spread_features, spread_masks):
        batch_size = spread_features.shape[0]
        outputs = []

        # Process global (macro) features once for all spreads
        global_context = self.global_fc(global_features)  # Shape: [batch_size, 16]

        for i in range(self.num_spreads):
            spread_input = spread_features[:, i, :]
            mask = spread_masks[:, i, :]
            spread_input = spread_input * mask  # Mask out padded values

            # Process spread-specific features
            spread_hidden = self.spread_fc(spread_input)

            # **NEW**: Concatenate macro/global features
            combined_input = torch.cat([spread_hidden, global_context], dim=1)

            # Predict final output for the spread
            spread_output = self.spread_heads[i](combined_input)
            outputs.append(spread_output.squeeze(-1))

        return torch.stack(outputs, dim=1)  # Shape: [batch_size, num_spreads]


def train_model(model, dataloader, num_epochs=20, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = torch.compile(model, mode="reduce-overhead")

    criterion = nn.MSELoss().to(device)  # ✅ Move loss to GPU
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=True)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    print(f"Number of batches: {len(dataloader)}")
    if len(dataloader) == 0:
        raise ValueError("Dataloader is empty! Check dataset loading.")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (global_features, spread_features, spread_masks, targets, spread_indices) in enumerate(dataloader):
            global_features, spread_features, spread_masks, targets = (
                global_features.to(device, non_blocking=True), spread_features.to(device, non_blocking=True),
                spread_masks.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            )

        # for batch_idx, (global_features, spread_features, spread_masks, targets, spread_indices) in enumerate(dataloader):
        #     spread_features, spread_masks, targets = (
        #         spread_features.to(device, non_blocking=True),
        #         spread_masks.to(device, non_blocking=True),
        #         targets.to(device, non_blocking=True)
        #     )

            optimizer.zero_grad()

            # with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            #     predictions = model(spread_features, spread_masks)
            #     loss = criterion(predictions, targets.to(device))
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                predictions = model(global_features, spread_features, spread_masks)  # 🚀 Runs in a different CUDA stream
                loss = criterion(predictions, targets)  # ✅ Compute loss in the same stream

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            # if batch_idx % 10 == 0:
            print(f"Epoch {epoch+1} | Batch {batch_idx+1}/{len(dataloader)} | Loss: {loss.item():.4f}")

        # scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}")

    print("Training complete!")


def predict(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    torch.cuda.empty_cache()  # ✅ Free GPU memory before inference
    all_predictions = []

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):  # ✅ Ensure bfloat16 on GPU
            for global_features, spread_features, spread_masks, targets, spread_indices in dataloader:
                global_features, spread_features, spread_masks = (
                    global_features.to(device), spread_features.to(device), spread_masks.to(device)
                )
                predictions = model(global_features, spread_features, spread_masks)
                all_predictions.append(predictions.cpu())

    return torch.cat(all_predictions, dim=0)
