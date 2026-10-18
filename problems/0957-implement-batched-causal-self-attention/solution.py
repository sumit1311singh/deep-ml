import numpy as np

def batched_causal_self_attention(X, W_query, W_key, W_value):
    """
    Batched causal self-attention.

    Args:
        X: nested list of shape (B, T, d_in)
        W_query, W_key, W_value: nested lists of shape (d_in, d_out)

    Returns:
        Nested list of shape (B, T, d_out) -- the context vectors.
    """
    # Your code here
    X, W_query, W_key, W_value = np.array(X), np.array(W_query), np.array(W_key), np.array(W_value)
    Q, K, V = X @ W_query, X @ W_key, X @ W_value

    B, T, d_k = K.shape

    attention_score = (Q @ K.transpose(0, 2, 1))/np.sqrt(d_k)

    mask = np.triu(np.full((T, T), -np.inf), k=1)
    attention_score = attention_score + mask

    shifted_attention_score = attention_score - np.max(attention_score, axis=-1, keepdims=True)

    exp_attention_score = np.exp(shifted_attention_score)
    attenion_weights = exp_attention_score/np.sum(exp_attention_score, axis=-1, keepdims=True)

    return attenion_weights @ V

