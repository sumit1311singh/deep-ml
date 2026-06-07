import numpy as np

def calculate_perplexity(probabilities: list[float]) -> float:
    """
    Calculate the perplexity of a language model given token probabilities.
    
    Args:
        probabilities: List of probabilities P(token_i | context) for each token
                      in the sequence, where each probability is in (0, 1]
    
    Returns:
        Perplexity value as a float
    """
    probs = np.array(probabilities)
    # Negative log likelihood per token
    nll = -np.log2(probs)
    # Average
    avg_nll = np.mean(nll)
    # Perplexity
    return float(2 ** avg_nll)