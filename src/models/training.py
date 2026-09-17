"""Common training policy: physical-unit MAE, validation-only checkpoint selection."""
import numpy as np
import torch


def fit_network(model, device, X_train, y_train, X_val, y_val, *, lr,
                weight_decay, epochs, batch_size, target_scale=None, patience=10):
    if epochs < 1 or batch_size < 1 or patience < 1:
        raise ValueError("epochs, batch_size and patience must be positive")
    model.to(device)
    # These small datasets fit in device memory; avoid a host transfer each batch.
    x = torch.as_tensor(X_train, dtype=torch.float32, device=device)
    y = torch.as_tensor(y_train, dtype=torch.float32, device=device)
    vx = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    vy = torch.as_tensor(y_val, dtype=torch.float32, device=device)
    scale = torch.as_tensor(1.0 if target_scale is None else target_scale,
                            dtype=torch.float32, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_epoch, updates, stale = float("inf"), 0, 0, 0
    best_weights = None
    for epoch in range(epochs):
        model.train()
        for indices in torch.randperm(len(x), device=device).split(batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = ((model(x[indices]) - y[indices]).abs() * scale).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            updates += 1
        model.eval()
        with torch.no_grad():
            val = float(((model(vx) - vy).abs() * scale).mean().item())
        if not np.isfinite(val):
            raise FloatingPointError("Nonfinite validation loss")
        if val < best:
            best, best_epoch, stale = val, epoch + 1, 0
            best_weights = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best_weights)
    model.eval()
    return {"epochs_requested": epochs, "epochs_executed": epoch + 1,
            "best_epoch": best_epoch, "optimizer_updates": updates,
            "batch_size": batch_size, "patience": patience,
            "selection_metric": "MAE_dB" if target_scale is not None else "MAE_scaled",
            "best_validation_loss": best}
