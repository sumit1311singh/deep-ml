import numpy as np


def split_and_baseline(X, y, train_frac, val_frac, test_frac, seed):
    """
    Seeded shuffle split of (X, y), fit a mean baseline on train, evaluate MAE on test.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
    y : np.ndarray, shape (n_samples,)
    train_frac, val_frac, test_frac : float
        Target fractions. Use int(n * frac) for train and val; remainder -> test.
    seed : int
        RNG seed for the shuffle.

    Returns
    -------
    mae : float
        Mean absolute error of the train-mean baseline on the test set.
    train_idx, val_idx, test_idx : np.ndarray
        1-D integer index arrays (a partition of range(n_samples)).
    """
    # TODO: permute indices with a seeded Generator
    # TODO: slice train / val / test index blocks
    # TODO: mu = mean of y[train_idx]; MAE of mu on y[test_idx]
    n = X.shape[0]
    
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    
    train_end = int(n*train_frac)
    val_end = train_end + int(n*val_frac)
    
    train_idx = idx[:train_end]
    val_idx   = idx[train_end:val_end]
    test_idx  = idx[val_end:]
    
    mu = np.mean(y[train_idx])
    
    mae = np.mean(np.abs(y[test_idx] - mu))
    
    return mae, train_idx, val_idx, test_idx
