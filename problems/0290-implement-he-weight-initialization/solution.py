import numpy as np

def he_initialization(n_in: int, n_out: int, mode: str = 'fan_in', distribution: str = 'normal', seed: int = None) -> np.ndarray:
    """
    Implement He (Kaiming) weight initialization.
    
    Parameters:
    n_in: number of input units
    n_out: number of output units
    mode: 'fan_in' or 'fan_out'
    distribution: 'normal' or 'uniform'
    seed: random seed for reproducibility
    
    Returns:
    numpy array of shape (n_in, n_out) with He-initialized weights
    """
    if seed is not None:
        np.random.seed(seed)

    # Select fan value
    if mode == 'fan_in':
        fan = n_in
    elif mode == 'fan_out':
        fan = n_out
    else:
        raise ValueError("mode must be 'fan_in' or 'fan_out'")

    if distribution == 'normal':
        std = np.sqrt(2.0 / fan)
        weights = np.random.randn(n_in, n_out) * std

    elif distribution == 'uniform':
        limit = np.sqrt(6.0 / fan)
        weights = np.random.uniform(-limit, limit, (n_in, n_out))

    else:
        raise ValueError("distribution must be 'normal' or 'uniform'")

    return weights