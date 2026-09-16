"""
Adaptive Spatio-Temporal Graph Neural Network (Adaptive STGNN) for LoRaWAN.

Formulation:
- Graph Topology: physical GPS graph blended with an adaptive learned graph.
- Spatial Modeling: Diffusion Graph Convolution aggregating cross-node radio/channel correlation.
- Temporal Modeling: 1D Causal Convolutions with residual gated activations.
- Residual Link: Directly anchored to the last observed RSSI (X_{t-1}).
"""

from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from src.node_topology import compute_physical_adjacency, GW_DISTANCES_M


class AdaptiveGraphConvolution(nn.Module):
    """
    Spatial Graph Convolution layer operating with a dynamically learned adjacency matrix.
    Computes diffusion: Z = A * X * W_spatial + X * W_self
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_nodes: int,
        physical_adjacency: np.ndarray,
        node_emb_dim: int = 8,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        # Two learnable node embeddings for source and target node interaction
        self.source_embedding = nn.Parameter(torch.randn(num_nodes, node_emb_dim))
        self.target_embedding = nn.Parameter(torch.randn(num_nodes, node_emb_dim))
        self.register_buffer(
            "physical_adjacency",
            torch.as_tensor(physical_adjacency, dtype=torch.float32),
        )
        # sigmoid(0) starts from an equal physical/adaptive mixture.
        self.adaptive_mix_logit = nn.Parameter(torch.tensor(0.0))

        self.w_spatial = nn.Linear(in_features, out_features, bias=False)
        self.w_self = nn.Linear(in_features, out_features, bias=True)
        self.layer_norm = nn.LayerNorm(out_features)

    def get_adjacency(self) -> torch.Tensor:
        """Returns the normalized learned adjacency matrix A in [0, 1]^(N x N)."""
        score = torch.mm(self.source_embedding, self.target_embedding.t())
        adaptive = F.softmax(F.relu(score), dim=-1)
        physical = self.physical_adjacency / self.physical_adjacency.sum(dim=-1, keepdim=True)
        adaptive_weight = torch.sigmoid(self.adaptive_mix_logit)
        return adaptive_weight * adaptive + (1.0 - adaptive_weight) * physical

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, num_nodes, in_features)
        Returns: (batch_size, num_nodes, out_features)
        """
        A = self.get_adjacency()  # (num_nodes, num_nodes)
        # Spatial diffusion: support = A @ x
        support = torch.einsum("nm, bmc -> bnc", A, x)
        out = self.w_spatial(support) + self.w_self(x)
        return self.layer_norm(F.relu(out))


class SpatioTemporalBlock(nn.Module):
    """
    Spatio-Temporal block combining:
    1. Temporal 1D Convolution
    2. Spatial Adaptive Graph Convolution
    3. Residual connection
    """
    def __init__(
        self,
        num_nodes: int,
        hidden_dim: int,
        physical_adjacency: np.ndarray,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        # Temporal convolution across time (kernel_size, causal padding)
        self.padding = kernel_size - 1
        self.temporal_conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=kernel_size,
            padding=self.padding
        )
        self.spatial_gcn = AdaptiveGraphConvolution(
            hidden_dim, hidden_dim, num_nodes, physical_adjacency
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, seq_len, num_nodes, hidden_dim)
        """
        batch_size, seq_len, num_nodes, feat_dim = x.shape

        # 1. Temporal processing per node
        # Reshape to (batch_size * num_nodes, hidden_dim, seq_len)
        x_t = x.permute(0, 2, 3, 1).contiguous().view(batch_size * num_nodes, feat_dim, seq_len)
        t_out = self.temporal_conv(x_t)
        if self.padding > 0:
            t_out = t_out[:, :, :-self.padding]
        t_out = F.relu(t_out)
        t_out = t_out.view(batch_size, num_nodes, feat_dim, seq_len).permute(0, 3, 1, 2)
        # t_out shape: (batch_size, seq_len, num_nodes, hidden_dim)

        # 2. Spatial processing across nodes for each time step
        # Reshape to (batch_size * seq_len, num_nodes, hidden_dim)
        s_in = t_out.contiguous().view(batch_size * seq_len, num_nodes, feat_dim)
        s_out = self.spatial_gcn(s_in)
        s_out = s_out.view(batch_size, seq_len, num_nodes, feat_dim)

        # 3. Residual connection
        out = self.norm(self.dropout(s_out) + x)
        return out


class AdaptiveSTGNN(nn.Module):
    """
    Complete Spatio-Temporal Graph Neural Network for LoRaWAN RSSI.
    """
    def __init__(
        self,
        n_targets: int = 8,
        n_exogenous: int = 4,
        seq_length: int = 24,
        pred_length: int = 6,
        hidden_dim: int = 32,
        num_blocks: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.n_targets = n_targets
        self.n_exogenous = n_exogenous
        self.seq_length = seq_length
        self.pred_length = pred_length

        # Per node: RSSI, weather variables, and normalized distance to the gateway.
        in_feat_per_node = 2 + n_exogenous
        self.input_proj = nn.Linear(in_feat_per_node, hidden_dim)

        physical_adjacency = compute_physical_adjacency()

        self.blocks = nn.ModuleList([
            SpatioTemporalBlock(
                num_nodes=n_targets,
                hidden_dim=hidden_dim,
                physical_adjacency=physical_adjacency,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])

        gateway_distance = GW_DISTANCES_M[:n_targets] / GW_DISTANCES_M[:n_targets].max()
        self.register_buffer(
            "gateway_distance",
            torch.as_tensor(gateway_distance, dtype=torch.float32).view(1, 1, n_targets, 1),
        )

        # Temporal pooling and projection head
        self.fc_head = nn.Sequential(
            nn.Linear(hidden_dim * seq_length, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_length)
        )

    def _prepare_spatio_temporal_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Converts flat (B, T, 8 + 4) tensor into spatio-temporal tensor (B, T, N=8, F=5).
        """
        batch_size, seq_len, _ = x.shape
        rssi = x[:, :, :self.n_targets]         # (B, T, 8)
        exog = x[:, :, self.n_targets:]         # (B, T, 4)

        # Expand exogenous across all nodes: (B, T, 8, 4)
        exog_expanded = exog.unsqueeze(2).repeat(1, 1, self.n_targets, 1)
        rssi_expanded = rssi.unsqueeze(-1)      # (B, T, 8, 1)

        distance = self.gateway_distance.expand(batch_size, seq_len, -1, -1)
        st_tensor = torch.cat([rssi_expanded, exog_expanded, distance], dim=-1)
        return st_tensor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, seq_length, in_features)
        Returns: (batch_size, pred_length, n_targets)
        """
        # Residual anchor from last known RSSI: (batch_size, 1, n_targets)
        last_rssi = x[:, -1:, :self.n_targets]

        st_tensor = self._prepare_spatio_temporal_tensor(x)  # (B, T, N, F)
        h = self.input_proj(st_tensor)                       # (B, T, N, hidden_dim)

        for block in self.blocks:
            h = block(h)

        # h shape: (B, T, N, hidden_dim) -> permute to (B, N, hidden_dim * T)
        batch_size, seq_len, num_nodes, hidden_dim = h.shape
        h_flat = h.permute(0, 2, 3, 1).contiguous().view(batch_size, num_nodes, hidden_dim * seq_len)

        # Predict delta per node: (B, N, pred_length)
        delta_per_node = self.fc_head(h_flat)
        # Permute to (B, pred_length, N)
        delta = delta_per_node.permute(0, 2, 1)

        return last_rssi + delta

    def get_learned_adjacency_matrix(self) -> np.ndarray:
        """Returns the spatial adjacency matrix learned by the first ST block."""
        with torch.no_grad():
            adj = self.blocks[0].spatial_gcn.get_adjacency()
            return adj.cpu().numpy()


class AdaptiveSTGNNTrainer:
    def __init__(
        self,
        n_targets: int = 8,
        n_exogenous: int = 4,
        seq_length: int = 24,
        pred_length: int = 6,
        hidden_dim: int = 24,
        num_blocks: int = 2,
        lr: float = 1e-3,
        dropout: float = 0.1,
        weight_decay: float = 1e-4,
        device: str = "cpu"
    ):
        self.device = torch.device(device)
        self.model = AdaptiveSTGNN(
            n_targets=n_targets,
            n_exogenous=n_exogenous,
            seq_length=seq_length,
            pred_length=pred_length,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks, dropout=dropout
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

    def get_learned_adjacency(self) -> np.ndarray:
        return self.model.get_learned_adjacency_matrix()
