import numpy as np

def train(X, y, W, b):
    """
    Train linear regression weights on standardized data.
    
    Args:
        X: numpy array of shape (n_samples, n_features) -- standardized features
        y: numpy array of shape (n_samples,) -- standardized targets
        W: numpy array of shape (n_features,) -- initial random weights
        b: float -- initial bias (0.0)
    
    Returns:
        W: numpy array of shape (n_features,) -- trained weights
        b: float -- trained bias
    """
    # TODO: implement your training strategy here
    # You can use ANY approach: gradient descent, normal equation,
    # momentum, adaptive learning rates, mini-batching, etc.
    X, y, W = np.array(X), np.array(y), np.array(W)

    m = X.shape[0]
    iters, lr= 1000, 0.01

    for _ in range(iters):
        y_hat = X @ W + b 

        error = y_hat - y

        w_g = (2/m)*np.dot(X.T, error)
        b_g = (2/m)*np.sum(error)

        W-=lr*w_g
        b-=lr*b_g

    return W, b    