import numpy as np

def min_max(x: list[float]) -> list[float]:
    """
    Perform Min-Max normalization to scale values to [0, 1].
    
    Args:
        x: A list of numerical values
    
    Returns:
        A new list with values normalized to [0, 1]
    """
    x=np.array(x)
    x_max, x_min = np.max(x), np.min(x)
    x_range = x_max - x_min
    return np.where(x_range==0, 1, (x-x_min)/x_range) 

