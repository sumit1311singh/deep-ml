import numpy as np

def sgd_update(X: np.ndarray, y: np.ndarray, weights: np.ndarray, learning_rate: float, n_iter: int) -> list:
    """
    Perform n_iter steps of stochastic gradient descent on a linear regression
    model with MSE loss, cycling through samples in order.

    Returns the final weight vector as a Python list.
    """
    n=len(y)
    for i in range(n_iter):
        X_i = X[i%n]

        y_hat = np.dot(X_i, weights)

        error = y_hat-y[i%n]
        gradient = 2*error*X_i

        weights = weights - learning_rate*gradient

    return weights
