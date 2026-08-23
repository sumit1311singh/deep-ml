import numpy as np

def simulate_clt(distribution: str, n: int, runs: int = 10000, seed: int = 42) -> dict:
    """
    Simulate the Central Limit Theorem.

    Args:
        distribution (str): The distribution to sample from ('uniform', 'exponential', 'bernoulli').
        n (int): Sample size.
        runs (int): Number of repeated experiments.
        seed (int): Random seed for reproducibility.

    Returns:
        dict: {'mean': float, 'std': float} of the standardized sample means.
    """
    np.random.seed(seed)
    
    # Draw samples according to the required exact calls
    if distribution == 'uniform':
        samples = np.random.uniform(0, 1, size=(runs, n))
        mu = 0.5
        sigma = np.sqrt(1/12)

    elif distribution == 'exponential':
        samples = np.random.exponential(1.0, size=(runs, n))
        mu = 1.0
        sigma = 1.0

    elif distribution == 'bernoulli':
        samples = (np.random.rand(runs, n) < 0.3).astype(float)
        mu = 0.3
        sigma = np.sqrt(0.3 * 0.7)

    else:
        raise ValueError("Unsupported distribution")

    # Sample means for each run
    sample_means = samples.mean(axis=1)

    # Standardize to Z-scores
    z_scores = (sample_means - mu) / (sigma / np.sqrt(n))

    # Return mean and std of standardized values
    return {
        'mean': float(np.round(z_scores.mean(), 3)),
        'std': float(np.round(z_scores.std(ddof=0), 3))
    }