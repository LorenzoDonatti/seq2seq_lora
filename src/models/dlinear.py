"""Original channel-independent NLinear/DLinear formulations, Zeng et al. (AAAI 2023).
Reference: https://github.com/cure-lab/LTSF-Linear/tree/main/models
RSSI channels only; shared temporal weights do not mix nodes or use weather.
"""
import torch
from torch import nn
from src.models.training import fit_network


class NLinear(nn.Module):
    def __init__(self, seq_len, pred_len, in_features, n_targets, individual=False):
        super().__init__()
        self.n_targets, self.individual = n_targets, individual
        self.linear = (nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(n_targets)])
                       if individual else nn.Linear(seq_len, pred_len))

    def forward(self, x):
        x = x[:, :, :self.n_targets]
        last = x[:, -1:, :].detach()
        centered = (x - last).transpose(1, 2)
        if self.individual:
            out = torch.stack([layer(centered[:, i]) for i, layer in enumerate(self.linear)], 1)
        else:
            out = self.linear(centered)
        return out.transpose(1, 2) + last


class DLinear(nn.Module):
    def __init__(self, seq_len, pred_len, in_features, n_targets, individual=False):
        super().__init__()
        self.n_targets, self.individual = n_targets, individual
        self.average = nn.AvgPool1d(25, stride=1)
        def projection():
            return (nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(n_targets)])
                    if individual else nn.Linear(seq_len, pred_len))
        self.seasonal, self.trend = projection(), projection()

    def forward(self, x):
        x = x[:, :, :self.n_targets].transpose(1, 2)
        padded = torch.cat([x[:, :, :1].expand(-1, -1, 12), x,
                            x[:, :, -1:].expand(-1, -1, 12)], dim=2)
        trend = self.average(padded)
        seasonal = x - trend
        if self.individual:
            out = torch.stack([self.seasonal[i](seasonal[:, i]) + self.trend[i](trend[:, i])
                               for i in range(self.n_targets)], 1)
        else:
            out = self.seasonal(seasonal) + self.trend(trend)
        return out.transpose(1, 2)


class NLinearTrainer:
    architecture = NLinear

    def __init__(self, seq_len, pred_len, in_features, n_targets, lr=1e-3,
                 weight_decay=1e-3, device="cpu", individual=False):
        self.device = torch.device(device)
        self.model = self.architecture(seq_len, pred_len, in_features, n_targets,
                                       individual=individual).to(self.device)
        self.lr, self.weight_decay = lr, weight_decay
        self.training_summary = {}

    def fit(self, X_train, y_train, X_val, y_val, epochs=64, batch_size=32,
            verbose=False, target_scale=None, patience=10):
        self.training_summary = fit_network(
            self.model, self.device, X_train, y_train, X_val, y_val,
            lr=self.lr, weight_decay=self.weight_decay, epochs=epochs,
            batch_size=batch_size, target_scale=target_scale, patience=patience)
        return self

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            return self.model(torch.as_tensor(X, dtype=torch.float32,
                                              device=self.device)).cpu().numpy()

    def total_parameters(self):
        return sum(p.numel() for p in self.model.parameters())


class DLinearTrainer(NLinearTrainer):
    architecture = DLinear
