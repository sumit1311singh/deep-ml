import numpy as np

def compute_norm(arr: np.ndarray, norm_type: str) -> float:
    """
    Compute the specified norm of the input array.
    
    Args:
        arr: Input numpy array (1D or 2D)
        norm_type: Type of norm ('l1', 'l2', or 'frobenius')
    
    Returns:
        The computed norm as a float
    """
    if norm_type == "l1":
        return sum(np.abs(arr))
    if norm_type == "l2":
        return np.sqrt(sum(arr**2))
    return np.round(np.sqrt(np.sum(arr**2)), 4)