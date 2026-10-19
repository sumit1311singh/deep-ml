import numpy as np

def mha_combined_qkv(x, W_qkv, W_out, num_heads):
    """
    Multi-head causal self-attention using a single combined QKV weight matrix.

    Args:
        x: array of shape (batch, num_tokens, d_in)
        W_qkv: array of shape (d_in, 3 * d_out)
        W_out: array of shape (d_out, d_out)
        num_heads: int, number of attention heads (d_out % num_heads == 0)

    Returns:
        Nested list of shape (batch, num_tokens, d_out).
    """
    B, T, _ = x.shape
    d_out, _ = W_out.shape

    head_dim = d_out//num_heads

    Q = x @ W_qkv[:, :d_out]
    K = x @ W_qkv[:, d_out:2*d_out]
    V = x @ W_qkv[:, 2*d_out:3*d_out]

    Q_heads = Q.reshape(B, T, num_heads, head_dim).transpose(0, 2, 1, 3)
    K_heads = K.reshape(B, T, num_heads, head_dim).transpose(0, 2, 1, 3)
    V_heads = V.reshape(B, T, num_heads, head_dim).transpose(0, 2, 1, 3)

    attention_score = (Q_heads @ K_heads.transpose(0, 1, 3, 2))/(np.sqrt(head_dim))

    mask = np.triu(np.full((T, T), -np.inf), k=1)
    attention_score = attention_score + mask

    shifted_attention_score = attention_score - np.max(attention_score, axis=-1, keepdims=True)

    exp_attention_score = np.exp(shifted_attention_score)
    attention_weights = exp_attention_score/np.sum(exp_attention_score, axis=-1, keepdims=True)

    attention_output = attention_weights @ V_heads
    attention_output = attention_output.transpose(0, 2, 1, 3)
    attention_output = attention_output.reshape(B, T, d_out)

    return attention_output @ W_out







