import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# -------- Paths --------
X_PATH = "../data/lstm_X.npy"
USERS_PATH = "../data/lstm_user_ids.csv"
OUT_DIR = "../outputs"
OUT_CSV = os.path.join(OUT_DIR, "lstm_anomaly_scores.csv")
os.makedirs(OUT_DIR, exist_ok=True)

# -------- Load data --------
X = np.load(X_PATH)  # (num_users, max_len, num_features)
users = pd.read_csv(USERS_PATH)["user_id"].tolist()

print("Loaded X shape:", X.shape)
print("Users:", len(users))

# -------- Fix sequence length (cap/truncate) --------
SEQ_LEN = 500
if X.shape[1] > SEQ_LEN:
    X = X[:, -SEQ_LEN:, :]
else:
    pad_len = SEQ_LEN - X.shape[1]
    X = np.pad(X, ((0, 0), (pad_len, 0), (0, 0)), mode="constant")

print("Using X shape:", X.shape)

# -------- Normalize features --------
# feature 0: event_type_encoded (integer-ish)
# feature 1: is_off_hours (0/1)
X = X.astype(np.float32)
max0 = X[:, :, 0].max()
if max0 > 0:
    X[:, :, 0] = X[:, :, 0] / max0

# -------- Torch setup --------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Device:", device)

X_t = torch.tensor(X, dtype=torch.float32)
dataset = TensorDataset(X_t)
loader = DataLoader(dataset, batch_size=8, shuffle=True)

# -------- Model: LSTM Autoencoder --------
class LSTMAE(nn.Module):
    def __init__(self, n_features, hidden1=64, hidden2=32):
        super().__init__()
        self.enc1 = nn.LSTM(input_size=n_features, hidden_size=hidden1, batch_first=True)
        self.enc2 = nn.LSTM(input_size=hidden1, hidden_size=hidden2, batch_first=True)

        self.dec1 = nn.LSTM(input_size=hidden2, hidden_size=hidden2, batch_first=True)
        self.dec2 = nn.LSTM(input_size=hidden2, hidden_size=hidden1, batch_first=True)

        self.out = nn.Linear(hidden1, n_features)

    def forward(self, x):
        # Encoder
        z, _ = self.enc1(x)
        z, _ = self.enc2(z)          # (B, T, hidden2)

        # Use last hidden state as embedding
        emb = z[:, -1, :]            # (B, hidden2)

        # Repeat embedding across time
        emb_rep = emb.unsqueeze(1).repeat(1, x.size(1), 1)  # (B, T, hidden2)

        # Decoder
        y, _ = self.dec1(emb_rep)
        y, _ = self.dec2(y)
        y = self.out(y)              # (B, T, n_features)
        return y

model = LSTMAE(n_features=X.shape[2]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# -------- Train --------
EPOCHS = 15
model.train()
for epoch in range(1, EPOCHS + 1):
    total_loss = 0.0
    for (batch,) in loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred, batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch.size(0)

    avg_loss = total_loss / len(dataset)
    print(f"Epoch {epoch:02d}/{EPOCHS} | loss={avg_loss:.6f}")

# -------- Reconstruction error per user --------
model.eval()
with torch.no_grad():
    X_in = X_t.to(device)
    X_pred = model(X_in).cpu().numpy()

mse = np.mean((X - X_pred) ** 2, axis=(1, 2))
score = (mse - mse.min()) / (mse.max() - mse.min() + 1e-9)

out = pd.DataFrame({
    "user_id": users,
    "lstm_recon_mse": mse,
    "lstm_score_norm": score
}).sort_values("lstm_score_norm", ascending=False)

out.to_csv(OUT_CSV, index=False)
print("Saved:", OUT_CSV)
print(out.head(10).to_string(index=False))
