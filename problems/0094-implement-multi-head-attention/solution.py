import numpy as np
from typing import Tuple

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Query, Key, and Value matrices.
    
    Args:
        X: Input matrix of shape (seq_len, d_model)
        W_q, W_k, W_v: Weight matrices of shape (d_model, d_model)
    
    Returns:
        Q, K, V matrices each of shape (seq_len, d_model)
    """
    # Your code here
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    return Q, K, V

def self_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_k)
    
    Returns:
        Attention output of shape (seq_len, d_k)
    """
    # Your code here
    d_k = K.shape[1]
    attention_score = (Q @ K.T)/np.sqrt(d_k)

    shifted_attention_score = attention_score - np.max(attention_score, axis=1, keepdims=True)

    exp_attention_score = np.exp(shifted_attention_score)
    softmax_attention_score = exp_attention_score/np.sum(exp_attention_score, axis=1, keepdims=True)

    return softmax_attention_score @ V

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, n_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    
    Args:
        Q, K, V: Matrices of shape (seq_len, d_model)
        n_heads: Number of attention heads
    
    Returns:
        Attention output of shape (seq_len, d_model)
    """
    # Your code here
    seq_len, d_model = Q.shape
    d_k = d_model//n_heads # remember to do '//' and not '/'

    Q_heads = Q.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2) # Q_heads -> (n_heads, seq_len, d_k)
    K_heads = K.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)
    V_heads = V.reshape(seq_len, n_heads, d_k).transpose(1, 0, 2)

    scores = []
    for head in range(n_heads):
        scores.append(self_attention(Q_heads[head], K_heads[head], V_heads[head]))

    return np.concatenate(scores, axis=1)





