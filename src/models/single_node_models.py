"""
Independent Single-Node models with Residual Connection.

Trains N separate dedicated models (one for each LoRa node).
Each model predicts the delta from its own last observed RSSI.
"""

from typing import List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


class SingleNodeLSTM(nn.Module):
    def __init__(self, in_features: int, hidden_dim: int, pred_length: int, num_layers: int = 1, dropout: float = 0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, pred_length)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, in_features)
        # Channel 0 is the node's own RSSI
        last_obs = x[:, -1, 0:1]  # (batch_size, 1)
        _, (h_n, _) = self.lstm(x)
        delta = self.fc(h_n[-1])  # (batch_size, pred_length)
        return last_obs + delta


class EnsembleSingleNodeLSTM:
    def __init__(
        self,
        n_targets: int = 8,
        hidden_dim: int = 24,
        pred_length: int = 1,
        num_layers: int = 1,
        lr: float = 1e-3,
        dropout: float = 0.1,
        weight_decay: float = 1e-4,
        device: str = "cpu"
    ):
        self.n_targets = n_targets
        self.pred_length = pred_length
        self.device = torch.device(device)
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.models: List[SingleNodeLSTM] = []
        self.training_summary = {}

    def _extract_node_features(self, X: np.ndarray, node_idx: int) -> np.ndarray:
        node_rssi = X[:, :, node_idx : node_idx + 1]
        exogenous = X[:, :, self.n_targets :]
        return np.concatenate([node_rssi, exogenous], axis=-1)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        epochs: int = 30,
        batch_size: int = 32,
        verbose: bool = False
    ):
        self.models = []
        criterion = nn.MSELoss()

        best_epochs = []
        updates = 0
        for node_i in range(self.n_targets):
            X_node_tr = self._extract_node_features(X_train, node_i)
            y_node_tr = y_train[:, :, node_i]

            X_node_val = self._extract_node_features(X_val, node_i)
            y_node_val = y_val[:, :, node_i]

            in_features = X_node_tr.shape[-1]
            model = SingleNodeLSTM(
                in_features=in_features,
                hidden_dim=self.hidden_dim,
                pred_length=self.pred_length,
                num_layers=self.num_layers,
                dropout=self.dropout,
            ).to(self.device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

            train_ds = TensorDataset(
                torch.tensor(X_node_tr, dtype=torch.float32),
                torch.tensor(y_node_tr, dtype=torch.float32)
            )
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

            best_val_loss = float("inf")
            best_weights = None

            for epoch in range(epochs):
                model.train()
                for bx, by in train_loader:
                    bx, by = bx.to(self.device), by.to(self.device)
                    optimizer.zero_grad()
                    pred = model(bx)
                    loss = criterion(pred, by)
                    loss.backward()
                    optimizer.step()
                    updates += 1

                model.eval()
                with torch.no_grad():
                    vx = torch.tensor(X_node_val, dtype=torch.float32).to(self.device)
                    vy = torch.tensor(y_node_val, dtype=torch.float32).to(self.device)
                    v_loss = criterion(model(vx), vy).item()
                    if v_loss < best_val_loss:
                        best_val_loss = v_loss
                        best_epoch = epoch + 1
                        best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            if best_weights is not None:
                model.load_state_dict(best_weights)
            model.eval()
            self.models.append(model)
            best_epochs.append(best_epoch)
        self.training_summary = {
            "epochs_requested": epochs,
            "best_epoch_per_node": best_epochs,
            "optimizer_updates": updates,
            "batch_size": batch_size,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        all_preds = []
        for node_i, model in enumerate(self.models):
            X_node = self._extract_node_features(X, node_i)
            with torch.no_grad():
                bx = torch.tensor(X_node, dtype=torch.float32).to(self.device)
                pred = model(bx).cpu().numpy()
                all_preds.append(pred)

        return np.stack(all_preds, axis=-1)

    def total_parameters(self) -> int:
        return sum(sum(p.numel() for p in m.parameters() if p.requires_grad) for m in self.models)
