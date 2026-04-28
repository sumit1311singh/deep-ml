import numpy as np

def xavier_init(fan_in: int, fan_out: int, mode: str = 'uniform', seed: int = 42) -> dict:
    """
    Perform Xavier/Glorot weight initialization.

    Args:
        fan_in (int): Number of input units.
        fan_out (int): Number of output units.
        mode (str): 'uniform' or 'normal'.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: Contains 'weights' (nested list), 'shape' (list), and 'param' (float).
    """
    np.random.seed(seed)

    if mode == 'uniform':
        limit = np.sqrt(6/(fan_in+fan_out))
        dictt = {
            'weights': np.round(np.random.uniform(-limit, limit, size=[fan_in, fan_out]), 4),
            'shape': [fan_in, fan_out],
            'param': np.round(limit, 4)
        } 
        return dictt
    elif mode == 'normal':
        limit = np.sqrt(2/(fan_in+fan_out))
        dictt = {
                'weights': np.round(np.random.normal(0, limit, size=[fan_in, fan_out]), 4),
                'shape': [fan_in, fan_out],
                'param': np.round(limit, 4)
            } 
        return dictt
    raise ValueError("mode must be 'uniform' or 'normal'")