import numpy as np

def map_estimate_bernoulli(observations: list, alpha: float, beta: float) -> float:
    """
    Compute the Maximum A Posteriori (MAP) estimate for a Bernoulli parameter.
    
    Args:
        observations: List of binary observations (0s and 1s)
        alpha: Alpha parameter of Beta prior (>= 1)
        beta: Beta parameter of Beta prior (>= 1)
    
    Returns:
        MAP estimate of the probability parameter, rounded to 4 decimal places
    """
    S = np.sum(observations)          # number of 1s
    F = len(observations) - S         # number of 0s

    a_post = S + alpha
    b_post = F + beta

    # MAP formula for Beta(a_post, b_post)
    map_est = (a_post - 1) / (a_post + b_post - 2)

    return float(np.round(map_est, 4))