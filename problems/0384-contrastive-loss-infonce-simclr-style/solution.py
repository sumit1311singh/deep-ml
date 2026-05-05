import numpy as np

def contrastive_loss(embeddings: np.ndarray, temperature: float) -> float:
    """
    Compute the NT-Xent (SimCLR-style) contrastive loss.
    
    Args:
        embeddings: Array of shape (2N, d) where consecutive pairs
                    (2i, 2i+1) are positive pairs.
        temperature: Temperature scaling parameter (tau > 0).
    
    Returns:
        The mean contrastive loss as a float.
    """
    z = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    sim = z @ z.T
    sim = sim / temperature

    np.fill_diagonal(sim, -np.inf)

    N = sim.shape[0]
    pos_idx = np.arange(N) ^ 1

    # log-sum-exp trick
    max_sim = np.max(sim, axis=1, keepdims=True)
    logsumexp = max_sim + np.log(np.sum(np.exp(sim - max_sim), axis=1, keepdims=True))
    
    loss = -sim[np.arange(N), pos_idx] + logsumexp.squeeze()
    
    return np.mean(loss)