import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    X, y = np.array(X), np.array(y)
    w = weights.copy()
    m = X.shape[0]

    for _ in range(n_epochs):

        # 🔀 Shuffle (real-world practice)
        #indices = np.random.permutation(m)
        #X, y = X[indices], y[indices]

        if method == 'batch':
            y_hat = X @ w
            dW = (2/m) * X.T @ (y_hat - y)
            w -= learning_rate * dW

        elif method == 'stochastic':
            for i in range(m):
                xi = X[i:i+1]
                yi = y[i:i+1]

                y_hat = xi @ w
                dW = 2 * xi.T @ (y_hat - yi)
                w -= learning_rate * dW

        elif method == 'mini_batch':
            for i in range(0, m, batch_size):
                Xb = X[i:i+batch_size]
                yb = y[i:i+batch_size]

                y_hat = Xb @ w
                dW = (2/Xb.shape[0]) * Xb.T @ (y_hat - yb)
                w -= learning_rate * dW

    return w
