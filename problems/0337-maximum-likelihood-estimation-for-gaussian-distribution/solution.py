import numpy as np

def gaussian_mle(data: np.ndarray) -> tuple:
    """
    Compute Maximum Likelihood Estimates for Gaussian distribution parameters.
    
    Args:
        data: 1D numpy array of observations
        
    Returns:
        Tuple of (mean_mle, variance_mle)
    """
    data_mean = np.mean(data)
    data_var = np.mean((data-data_mean)**2)
    return (data_mean, data_var)