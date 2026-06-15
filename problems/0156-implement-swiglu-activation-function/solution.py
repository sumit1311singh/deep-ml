import numpy as np

def SwiGLU(x: np.ndarray) -> np.ndarray:
    """
    SwiGLU activation function.
    
    Args:
        x: np.ndarray of shape (batch_size, 2d)
    Returns:
        np.ndarray of shape (batch_size, d)
    """
    # Split into two halves
    d = x.shape[1] // 2
    x1, x2 = x[:, :d], x[:, d:]
    
    # Swish activation: x * sigmoid(x)
    swish = x2 * (1 / (1 + np.exp(-x2)))
    
    # SwiGLU: swish(x1) * x2
    return x1 * swish 
