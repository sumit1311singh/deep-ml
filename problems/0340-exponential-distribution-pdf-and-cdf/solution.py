import numpy as np

def exponential_distribution(x: list, lam: float) -> dict:
    """
    Compute exponential distribution properties.
    
    Args:
        x: Points at which to evaluate PDF and CDF
        lam: Rate parameter (lambda) of the distribution
        
    Returns:
        Dictionary with 'pdf', 'cdf', 'mean', and 'variance' keys
    """
    if lam<=0:
        dictt = {
            'pdf': None,
            'cdf': None,
            'mean': None,
            'variance': None
        }
        return dictt
    x=np.array(x)
    pdf = np.round(np.where(x>=0, lam*np.exp(-lam*x), 0), 4).tolist()
    cdf = np.round(np.where(x>=0, 1 - np.exp(-lam*x), 0), 4).tolist()
    mean = np.round(1/lam, 4)
    variance = np.round(1/(lam**2), 4)
    dictt = {
        'pdf': pdf,
        'cdf': cdf,
        'mean': mean,
        'variance': variance
    }
    return dictt