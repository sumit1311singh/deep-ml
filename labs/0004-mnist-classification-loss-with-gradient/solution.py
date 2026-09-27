import numpy as np

def loss_function(preds: np.ndarray, target: np.ndarray, reduction: str = "mean", **kwargs):
    """
    preds:     [N, C] softmax probabilities (rows sum to 1)
    target:    [N]    class indices (int64)
    reduction: how to aggregate per-sample losses:
               "mean" → average over batch (gradient scaled by 1/N)
               "sum"  → sum over batch (gradient unscaled)
               "none" → return per-sample loss vector (no aggregation)
    **kwargs:  absorbs any extra arguments from the training harness

    Returns: (loss, grad) where grad has the same shape as preds
    """
    # Your implementation here
    N, C = preds.shape
    T = np.eye(C)[target]

    p = np.clip(preds, 1e-12, 1.0)

    per_sample = -np.log(p[np.arange(N), target])
    if reduction == "mean":
        loss = per_sample.mean()
    elif reduction == "sum":
        loss = per_sample.sum()
    else:
        loss = per_sample

    grad = np.zeros_like(preds)
    grad[np.arange(N), target] = -1 / p[np.arange(N), target]

    if reduction == "mean":
        grad/=N

    return loss, grad