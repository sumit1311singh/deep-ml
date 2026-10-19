import numpy as np

def causal_mask_attention(attn_weights: np.ndarray, method: str = 'tril') -> list:
    """Apply causal masking two ways and return the resulting attention matrix as a nested list."""
    T, _ = attn_weights.shape

    if method == 'tril':
        # Approach 1: Multiply by a lower-triangular mask
        mask = np.tril(np.ones((T, T)))
        masked_weights = attn_weights * mask

        # Re-normalize each row
        row_sums = np.sum(masked_weights, axis=-1, keepdims=True)
        result = masked_weights / row_sums

    elif method == 'triu':
        # Approach 2: Reconstruct scores from probabilities
        scores = np.log(attn_weights)

        # Mask positions strictly above the main diagonal
        mask = np.triu(np.ones((T, T), dtype=bool), k=1)
        scores[mask] = -np.inf

        # Stable softmax along the last axis
        shifted_scores = scores - np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(shifted_scores)
        result = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    return result.tolist()