"""Independent per-node copies of exactly the same neural architecture."""
import numpy as np


class DedicatedNodeTrainer:
    def __init__(self, trainer_class, n_targets, trainer_kwargs, include_weather=True):
        self.trainer_class = trainer_class
        self.n_targets = n_targets
        self.trainer_kwargs = trainer_kwargs
        self.include_weather = include_weather
        self.models = []
        self.training_summary = {}

    def features(self, X, node):
        own = X[:, :, node:node+1]
        return np.concatenate([own, X[:, :, self.n_targets:]], axis=-1) if self.include_weather else own

    def fit(self, X_train, y_train, X_val, y_val, epochs=64, batch_size=32,
            verbose=False, target_scale=None, patience=10):
        self.models = []
        summaries = []
        for node in range(self.n_targets):
            model = self.trainer_class(**self.trainer_kwargs)
            model.fit(self.features(X_train, node), y_train[:, :, node:node+1],
                      self.features(X_val, node), y_val[:, :, node:node+1],
                      epochs=epochs, batch_size=batch_size, patience=patience,
                      target_scale=None if target_scale is None else target_scale[node:node+1])
            self.models.append(model)
            summaries.append(model.training_summary)
        self.training_summary = {"per_node": summaries, "epochs_requested": epochs,
                                 "best_epoch_per_node": [s["best_epoch"] for s in summaries],
                                 "optimizer_updates": sum(s["optimizer_updates"] for s in summaries)}
        return self

    def predict(self, X):
        return np.concatenate([m.predict(self.features(X, i)) for i, m in enumerate(self.models)],axis=-1)

    def total_parameters(self):
        return sum(m.total_parameters() for m in self.models)
