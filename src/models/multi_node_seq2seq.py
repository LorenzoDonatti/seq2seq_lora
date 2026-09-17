"""
Integrated recurrent encoder-decoder with additive temporal attention.

LSTM encoder and autoregressive LSTMCell decoder, following the encoder-decoder
and additive-attention pattern of Sutskever et al. (2014) and Bahdanau et al.
(2015), adapted to continuous multi-node RSSI. Free-running training uses previous
predictions, never future observations; a last-observation residual anchors outputs.

Incorporates a direct residual link from the last observed time step (X_{t-1}),
allowing the neural network to focus exclusively on learning the dynamic
variations (Delta RSSI) rather than re-learning the static DC offset.
"""

import torch
import torch.nn as nn
import numpy as np
from src.models.training import fit_network


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

        # Recurrent attention decoder, initialized with the encoder state.
        self.decoder = nn.LSTMCell(n_targets + hidden_dim, hidden_dim)
        self.decoder_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_targets)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, seq_len, in_features)
        last_obs = x[:, -1:, :self.n_targets]  # Residual base: (batch_size, 1, n_targets)

        encoder_outputs, (h_n, c_n) = self.encoder(x)
        hidden, cell = h_n[-1], c_n[-1]
        anchor = last_obs[:, 0]
        previous = anchor
        predictions = []
        for _ in range(self.pred_length):
            context = self.attention(hidden, encoder_outputs)
            hidden, cell = self.decoder(torch.cat([previous, context], dim=-1),
                                        (hidden, cell))
            previous = anchor + self.decoder_head(torch.cat([hidden, context], dim=-1))
            predictions.append(previous)
        return torch.stack(predictions, dim=1)


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

    def fit(self, X_train, y_train, X_val, y_val, epochs=64,
            batch_size=32, verbose=False, target_scale=None, patience=10):
        self.training_summary = fit_network(
            self.model, self.device, X_train, y_train, X_val, y_val,
            lr=self.lr, weight_decay=self.weight_decay, epochs=epochs,
            batch_size=batch_size, target_scale=target_scale, patience=patience)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            bx = torch.tensor(X, dtype=torch.float32).to(self.device)
            return self.model(bx).cpu().numpy()

    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
