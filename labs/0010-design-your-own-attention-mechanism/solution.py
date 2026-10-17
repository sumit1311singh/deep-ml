import numpy as np

def attention(Q, K, V):
    """
    Compute attention over a sequence.
    
    Args:
        Q: Query matrix, shape (batch_size, query_len, dim)
        K: Key matrix, shape (batch_size, key_len, dim)
        V: Value matrix, shape (batch_size, key_len, dim)
    
    Returns:
        output: Attended values, shape (batch_size, query_len, dim)
    
    The attention mechanism should:
    1. Compute compatibility between queries and keys
    2. Convert to attention weights (non-negative, sum to 1)
    3. Use weights to compute weighted sum of values
    """
    batch_size, query_len, dim = Q.shape
    _, key_len, _ = K.shape
    
    # Your implementation here
    # Hint: Think about how queries "ask questions" and keys "provide answers"
    
    attention_score = (Q @ K.transpose(0, 2, 1))/np.sqrt(dim)

    shifted_attention_score = attention_score - np.max(attention_score, axis=-1, keepdims=True)

    exp_attention_score = np.exp(shifted_attention_score)
    softmax_attention_score = exp_attention_score/np.sum(exp_attention_score, axis=-1, keepdims=True)

    return softmax_attention_score @ V
