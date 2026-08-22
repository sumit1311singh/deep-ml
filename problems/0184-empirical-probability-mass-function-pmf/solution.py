import numpy as np

def empirical_pmf(samples):
    """
    Given an iterable of integer samples, return a list of (value, probability)
    pairs sorted by value ascending.
    """
    samples = np.array(samples)
    if len(samples)==0:
        return []
    values, counts = np.unique(samples, return_counts=True)
    ans = list(zip(values, counts / len(samples)))
    return sorted(ans)