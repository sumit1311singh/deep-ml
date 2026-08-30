import numpy as np

def standard_scaler(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    Fit a standard scaler on X_train and transform X_test.
    Returns the standardized X_test as a numpy array.
    """
    X_train_mean = np.mean(X_train, axis=0, keepdims=True)
    X_train_std = np.std(X_train, axis=0, keepdims=True)
    mask=(X_train_std==0)
    X_train_std[mask]=1
    #print(X_train.shape, X_test.shape, X_train_mean.shape, X_train_std.shape)
    #return [[-0.6124, -0.6124], [0.6124, 0.6124]]
    return (X_test-X_train_mean)/X_train_std
