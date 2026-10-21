import numpy as np

def scaled_attention_weights(Q: np.ndarray, K: np.ndarray) -> list:
    """
    Compute scaled dot-product attention weights.

    Args:
        Q: (n_q, d_k) query matrix
        K: (n_k, d_k) key matrix

    Returns:
        Attention weights of shape (n_q, n_k) as a nested list,
        each entry rounded to 4 decimal places.
    """
    _, d_k = K.shape

    attention_scores = (Q @ K.T)/np.sqrt(d_k)

    stable_attention_scores = attention_scores - np.max(attention_scores, axis=-1, keepdims=True)

    exp_attention_scores = np.exp(stable_attention_scores)
    scaled_attention_weights = exp_attention_scores/np.sum(exp_attention_scores, axis=-1, keepdims=True)

    return scaled_attention_weights
