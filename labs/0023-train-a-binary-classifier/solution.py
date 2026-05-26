import numpy as np
# You can import any sklearn module you need

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train(X_train, y_train, X_val, y_val):
    """
    Train a binary classifier.
    
    Args:
        X_train: numpy array of shape (n_samples, 30) -- standardized features
        y_train: numpy array of shape (n_samples,) -- binary labels (0 or 1)
        X_val:   numpy array of shape (n_val, 30) -- standardized
        y_val:   numpy array of shape (n_val,) -- validation labels
    
    Returns:
        predict: callable that takes X (n, 30) and returns y_pred (n,) of 0s and 1s
    """
    # TODO: train a model and return a predict function
    
    W, b = np.zeros(X_train.shape[1]), 0
    iters, lr = 1000, 0.01
    m=X_train.shape[0]

    for _ in range(iters):
        y_hat = X_train @ W + b 
        y_pred = sigmoid(y_hat)

        error = y_pred - y_train

        w_grad = (1/m) * np.dot(X_train.T, error)
        b_grad = (1/m) * np.sum(error)

        W-=lr*w_grad
        b-=lr*b_grad

    def predict(X):
        y_hat = X @ W + b 
        y_pred = sigmoid(y_hat)
        return (y_pred>=0.5).astype(int)

    return predict 
