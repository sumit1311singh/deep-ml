import numpy as np

def group_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, num_groups: int, epsilon: float = 1e-5) -> np.ndarray:

    B, C, H, W = X.shape

    G = num_groups
    X = X.reshape(B, G, C//G, H, W)

    x_mean = np.mean(X, axis=(2, 3, 4), keepdims=True)
    x_var = np.var(X, axis=(2, 3, 4), keepdims=True)

    x_hat = (X - x_mean)/np.sqrt(x_var + epsilon)

    x_hat = x_hat.reshape(B, C, H, W)
    y = gamma * x_hat + beta

    return y