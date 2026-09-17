"""
Adaptive Spatio-Temporal Graph Neural Network (Adaptive STGNN) for LoRaWAN.

Formulation:
- Graph Topology: geographic distance prior blended with a static learned graph.
- Ablations: no cross-node edges, geographic-only, learned-only, and hybrid.
- Adjacencies represent predictive weights, not physical radio connectivity.
- Spatial Modeling: Diffusion Graph Convolution aggregating cross-node radio/channel correlation.
- Temporal Modeling: gated, dilated causal convolutions; residual connections.
- Residual Link: Directly anchored to the last observed RSSI (X_{t-1}).
"""

from typing import Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from src.models.training import fit_network

from src.node_topology import (
    compute_physical_adjacency, haversine_distance, NODE_COORDS, GATEWAY_COORDS,
    GW_DISTANCES_M,
)


class AdaptiveGraphConvolution(nn.Module):
    """
    Graph convolution with a static adjacency learned during training.
    Computes diffusion: Z = A * X * W_spatial + X * W_self
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_nodes: int,
        physical_adjacency: np.ndarray,
        node_emb_dim: int = 4,
        graph_mode: str = "hybrid",
    ):
        super().__init__()
        self.num_nodes = num_nodes
        if graph_mode not in ("none", "physical", "adaptive", "hybrid"):
            raise ValueError("Unknown graph mode")
        if physical_adjacency.shape != (num_nodes, num_nodes):
            raise ValueError("Graph dimensions must match target node ordering")
        self.graph_mode = graph_mode
        self.register_buffer("off_diagonal", ~torch.eye(num_nodes, dtype=torch.bool))
        physical = torch.as_tensor(physical_adjacency, dtype=torch.float32).clone()
        physical.fill_diagonal_(0)
        physical = physical / physical.sum(-1, keepdim=True).clamp_min(1e-12)
        self.register_buffer("physical_adjacency", physical)
        if graph_mode in ("adaptive", "hybrid"):
            self.source_embedding = nn.Parameter(torch.randn(num_nodes, node_emb_dim)*0.1)
            self.target_embedding = nn.Parameter(torch.randn(num_nodes, node_emb_dim)*0.1)
        if graph_mode == "hybrid":
            self.adaptive_mix_logit = nn.Parameter(torch.tensor(0.0))
        self.w_spatial = (nn.Linear(in_features, out_features, bias=False)
                          if graph_mode != "none" else None)
        self.w_self = nn.Linear(in_features, out_features, bias=True)
        self.layer_norm = nn.LayerNorm(out_features)

    def get_adjacency(self) -> torch.Tensor:
        """Returns the normalized learned adjacency matrix A in [0, 1]^(N x N)."""
        if self.graph_mode == "none" or self.num_nodes == 1:
            return torch.zeros_like(self.physical_adjacency)
        if self.graph_mode == "physical":
            return self.physical_adjacency
        score = torch.mm(self.source_embedding, self.target_embedding.t())
        score = score.masked_fill(~self.off_diagonal, -torch.inf)
        adaptive = F.softmax(score, dim=-1)
        if self.graph_mode == "adaptive":
            return adaptive
        weight = torch.sigmoid(self.adaptive_mix_logit)
        return weight * adaptive + (1-weight) * self.physical_adjacency

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (batch_size, num_nodes, in_features)
        Returns: (batch_size, num_nodes, out_features)
        """
        A = self.get_adjacency()  # (num_nodes, num_nodes)
        # Spatial diffusion: support = A @ x
        support = torch.einsum("nm, bmc -> bnc", A, x)
        out = self.w_self(x)
        if self.w_spatial is not None:
            out = out + self.w_spatial(support)
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
        dilation: int = 1,
        graph_mode: str = "hybrid",
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        # Temporal convolution across time (kernel_size, causal padding)
        self.padding = (kernel_size - 1) * dilation
        self.temporal_conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=2 * hidden_dim,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=self.padding
        )
        self.spatial_gcn = AdaptiveGraphConvolution(
            hidden_dim, hidden_dim, num_nodes, physical_adjacency, graph_mode=graph_mode
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
        value, gate = t_out.chunk(2, dim=1)
        t_out = torch.tanh(value) * torch.sigmoid(gate)
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
        dropout: float = 0.1,
        graph_mode: str = "hybrid",
        node_coordinates: Optional[np.ndarray] = None,
        gateway_coordinates: Optional[np.ndarray] = None,
        gateway_distances_m: Optional[np.ndarray] = None,
    ):
        super().__init__()
        self.n_targets = n_targets
        self.n_exogenous = n_exogenous
        self.seq_length = seq_length
        self.pred_length = pred_length

        # Per node: RSSI, weather variables, and normalized distance to the gateway.
        in_feat_per_node = 2 + n_exogenous
        self.input_proj = nn.Linear(in_feat_per_node, hidden_dim)

        using_default_topology = node_coordinates is None
        node_coordinates = np.asarray(
            NODE_COORDS[:n_targets] if using_default_topology else node_coordinates,
            dtype=np.float64,
        )
        gateway_coordinates = np.asarray(
            GATEWAY_COORDS if gateway_coordinates is None else gateway_coordinates, dtype=np.float64
        )
        if node_coordinates.shape != (n_targets, 2) or gateway_coordinates.shape != (2,) or num_blocks < 1:
            raise ValueError("Invalid node count or number of graph blocks")
        physical_adjacency = compute_physical_adjacency(node_coords=node_coordinates)

        self.blocks = nn.ModuleList([
            SpatioTemporalBlock(
                num_nodes=n_targets,
                hidden_dim=hidden_dim,
                physical_adjacency=physical_adjacency,
                dropout=dropout, dilation=2**block_index, graph_mode=graph_mode,
            )
            for block_index in range(num_blocks)
        ])

        if gateway_distances_m is None and using_default_topology:
            gateway_distances_m = GW_DISTANCES_M[:n_targets]
        gateway_distance = np.asarray(
            ([haversine_distance(lat, lon, gateway_coordinates[0], gateway_coordinates[1])
              for lat, lon in node_coordinates]
             if gateway_distances_m is None else gateway_distances_m),
            dtype=np.float64,
        )
        if gateway_distance.shape != (n_targets,) or not np.all(gateway_distance > 0):
            raise ValueError("Invalid gateway distances")
        gateway_distance = gateway_distance / gateway_distance.max()
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
        Converts flat (B, T, 8 + 4) tensor into spatio-temporal tensor (B, T, N, F=2+n_exogenous).
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
        device: str = "cpu",
        graph_mode: str = "hybrid",
        node_coordinates: Optional[np.ndarray] = None,
        gateway_coordinates: Optional[np.ndarray] = None,
        gateway_distances_m: Optional[np.ndarray] = None,
    ):
        self.device = torch.device(device)
        self.model = AdaptiveSTGNN(
            n_targets=n_targets,
            n_exogenous=n_exogenous,
            seq_length=seq_length,
            pred_length=pred_length,
            hidden_dim=hidden_dim,
            num_blocks=num_blocks, dropout=dropout, graph_mode=graph_mode,
            node_coordinates=node_coordinates, gateway_coordinates=gateway_coordinates,
            gateway_distances_m=gateway_distances_m,
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

    def get_learned_adjacency(self) -> np.ndarray:
        return self.model.get_learned_adjacency_matrix()

    def graph_diagnostics(self):
        """All layers, rather than only the first adjacency plot."""
        return {
            "edge_orientation": "A[receiver, sender]",
            "learned_graph": "static after fitting; not time-varying or causal connectivity",
            "temporal_receptive_field": 1 + 2 * (2**len(self.model.blocks)-1),
            "readout": "uses all observed temporal features",
            "layers": [
                {"mode": block.spatial_gcn.graph_mode,
                 "adjacency": block.spatial_gcn.get_adjacency().detach().cpu().tolist(),
                 "adaptive_weight": float(torch.sigmoid(block.spatial_gcn.adaptive_mix_logit).item())
                    if hasattr(block.spatial_gcn, "adaptive_mix_logit") else None}
                for block in self.model.blocks
            ],
        }
