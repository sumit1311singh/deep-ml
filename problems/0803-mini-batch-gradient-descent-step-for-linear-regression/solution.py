import numpy as np

def mini_batch_gd_step(X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float, batch_indices: list, lr: float) -> np.ndarray:
    """
    Perform one mini-batch gradient descent update step for linear regression with MSE loss.
    Returns a 1D array of length D+1: updated weights followed by updated bias.
    """
    m=len(batch_indices)

    X_batch = X[batch_indices]
    y_batch = y[batch_indices]

    y_hat = np.dot(X_batch, weights) + bias

    error = y_hat-y_batch
    w_gradient = (2/m)*np.dot(X_batch.T, error)
    b_gradient = (2/m)*np.sum(error)

    weights -= lr*w_gradient
    bias -= lr*b_gradient

    result = np.append(weights, bias)

    return result