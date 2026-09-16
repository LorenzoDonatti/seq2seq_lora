"""
Integrated Multi-Node Seq2Seq with Temporal Attention and Residual Baseline Connection.

Incorporates a direct residual link from the last observed time step (X_{t-1}),
allowing the neural network to focus exclusively on learning the dynamic
variations (Delta RSSI) rather than re-learning the static DC offset.
"""

from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        seq_len = keys.size(1)
        q_expanded = query.unsqueeze(1).repeat(1, seq_len, 1)
        combined = torch.cat((q_expanded, keys), dim=2)
        energy = torch.tanh(self.attn(combined))
        scores = self.v(energy).squeeze(2)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), keys).squeeze(1)
        return context


class MultiNodeSeq2SeqAttention(nn.Module):
    def __init__(
        self,
        in_features: int,
        n_targets: int,
        pred_length: int,
        hidden_dim: int = 32,
        num_layers: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.in_features = in_features
        self.n_targets = n_targets
        self.pred_length = pred_length
        self.hidden_dim = hidden_dim

        # Encoder
        self.encoder = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Attention
        self.attention = TemporalAttention(hidden_dim)

        # Multi-Horizon Delta Projection Head
        self.decoder_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_length * n_targets)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, in_features)
        last_obs = x[:, -1:, :self.n_targets]  # Residual base: (batch_size, 1, n_targets)

        encoder_outputs, (h_n, _) = self.encoder(x)
        last_hidden = h_n[-1]

        context = self.attention(last_hidden, encoder_outputs)
        rep = torch.cat([last_hidden, context], dim=-1)

        delta = self.decoder_head(rep).view(-1, self.pred_length, self.n_targets)
        # Final prediction is Last Observation + Dynamic Delta
        return last_obs + delta


class MultiNodeSeq2SeqTrainer:
    def __init__(
        self,
        in_features: int,
        n_targets: int,
        pred_length: int,
        hidden_dim: int = 32,
        num_layers: int = 1,
        lr: float = 1e-3,
        dropout: float = 0.1,
        weight_decay: float = 1e-4,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.model = MultiNodeSeq2SeqAttention(
            in_features=in_features,
            n_targets=n_targets,
            pred_length=pred_length,
            hidden_dim=hidden_dim,
            num_layers=num_layers, dropout=dropout
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
        epochs: int = 35,
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
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
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
