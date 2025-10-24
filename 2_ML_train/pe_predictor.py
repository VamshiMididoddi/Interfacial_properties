import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np

# =====================================================
# 1. Load Dataset
# =====================================================
data = pd.read_csv("md_simulation_data.csv")

# Columns: step temp press KE PE Energy Vol
X = data[["temp", "Vol"]].values
y = data["PE"].values.reshape(-1, 1)

# =====================================================
# 2. Normalize features and target
# =====================================================
x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_scaled = x_scaler.fit_transform(X)
y_scaled = y_scaler.fit_transform(y)

X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

# =====================================================
# 3. Train-test split
# =====================================================
dataset = TensorDataset(X_tensor, y_tensor)
n_total = len(dataset)
n_train = int(0.8 * n_total)
n_val = n_total - n_train
train_ds, val_ds = random_split(dataset, [n_train, n_val])

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

# =====================================================
# 4. Define Neural Network
# =====================================================
class PE_Predictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

model = PE_Predictor()

# =====================================================
# 5. Training setup
# =====================================================
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
epochs = 300
patience = 20  # for early stopping

best_val_loss = float("inf")
patience_counter = 0

# =====================================================
# 6. Training Loop
# =====================================================
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
    train_loss /= n_train

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            val_loss += loss.item() * xb.size(0)
    val_loss /= n_val

    print(f"Epoch {epoch+1:03d}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")

    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_pe_model.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# =====================================================
# 7. Evaluation and Plot
# =====================================================
# Load best model
model.load_state_dict(torch.load("best_pe_model.pth"))
model.eval()

# Predict for all data
with torch.no_grad():
    y_pred_scaled = model(X_tensor).numpy()

# Inverse transform
y_pred = y_scaler.inverse_transform(y_pred_scaled)
y_true = y

# Compute R² and RMSE
r2 = 1 - np.sum((y_true - y_pred)**2) / np.sum((y_true - np.mean(y_true))**2)
rmse = np.sqrt(np.mean((y_true - y_pred)**2))
print(f"\nR²: {r2:.4f}, RMSE: {rmse:.4f} eV")

# =====================================================
# 8. Plot results
# =====================================================
plt.figure(figsize=(6,6))
plt.scatter(y_true, y_pred, c='blue', alpha=0.7)
plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
plt.xlabel("True PE (eV)")
plt.ylabel("Predicted PE (eV)")
plt.title(f"PE Prediction (R² = {r2:.3f}, RMSE = {rmse:.3f} eV)")
plt.tight_layout()
plt.savefig("pe_prediction_plot.png", dpi=200)
plt.show()
