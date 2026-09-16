"""
NLinear (AAAI 2023: 'Are Transformers Effective for Time Series?').

Normalizes the input by subtracting the last observed value (X_{t-1}),
maps the trend and seasonal variations using linear projection,
and adds back the last observed value.

Solves distribution shifts and prevents neural networks from drifting away from
the baseline in highly autocorrelated series.
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


class NLinear(nn.Module):
    def __init__(self, seq_len: int, pred_len: int, in_features: int, n_targets: int):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.in_features = in_features
        self.n_targets = n_targets

        # Maps normalized input sequence to prediction delta
        self.linear = nn.Linear(seq_len * in_features, pred_len * n_targets)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, in_features)
        last_val = x[:, -1:, :self.n_targets]  # (batch, 1, n_targets)

        # Normalize target features by subtracting last known observation
        x_norm = x.clone()
        x_norm[:, :, :self.n_targets] = x_norm[:, :, :self.n_targets] - last_val

        batch_size = x.size(0)
        flat_x = x_norm.contiguous().view(batch_size, -1)
        delta = self.linear(flat_x).view(batch_size, self.pred_len, self.n_targets)

        # Add back the last observation
        return last_val + delta


class NLinearTrainer:
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        in_features: int,
        n_targets: int,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.model = NLinear(
            seq_len=seq_len,
            pred_len=pred_len,
            in_features=in_features,
            n_targets=n_targets
        ).to(self.device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.training_summary = {}

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 40,
        batch_size: int = 32,
        verbose: bool = False
    ):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()

        train_ds = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        best_val_loss = float("inf")
        best_weights = None
        best_epoch = 0
        updates = 0

        for epoch in range(epochs):
            self.model.train()
            for bx, by in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                pred = self.model(bx)
                loss = criterion(pred, by)
                loss.backward()
                optimizer.step()
                updates += 1

            self.model.eval()
            with torch.no_grad():
                vx = torch.tensor(X_val, dtype=torch.float32).to(self.device)
                vy = torch.tensor(y_val, dtype=torch.float32).to(self.device)
                v_loss = criterion(self.model(vx), vy).item()
                if v_loss < best_val_loss:
                    best_val_loss = v_loss
                    best_epoch = epoch + 1
                    best_weights = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

        if best_weights is not None:
            self.model.load_state_dict(best_weights)
        self.model.eval()
        self.training_summary = {"epochs_requested": epochs, "best_epoch": best_epoch,
                                 "optimizer_updates": updates, "batch_size": batch_size,
                                 "best_val_mse_scaled": best_val_loss}

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            bx = torch.tensor(X, dtype=torch.float32).to(self.device)
            return self.model(bx).cpu().numpy()

    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


# Backward-compatible import for older notebooks.
DLinearTrainer = NLinearTrainer
