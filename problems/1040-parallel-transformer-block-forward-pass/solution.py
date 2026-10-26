import numpy as np

def parallel_transformer_block(x, gamma, beta, W_qkv, W_o, W1, W2, eps=1e-5):
    """
    Parallel transformer block: y = x + Attn(LN(x)) + FFN(LN(x)).
    Single-head self-attention, ReLU FFN, no biases.
    Returns a numpy array of shape (seq_len, d_model).
    """
    x_mean = np.mean(x, axis=-1, keepdims=True)
    x_var = np.var(x, axis=-1, keepdims=True)
    
    lnorm_x = gamma * (x-x_mean)/np.sqrt(x_var+eps) + beta

    W_qkv = np.asarray(W_qkv)
    d_model = W_qkv.shape[0]
    Q = lnorm_x @ W_qkv[:, :d_model]
    K = lnorm_x @ W_qkv[:, d_model:2 * d_model]
    V = lnorm_x @ W_qkv[:, 2 * d_model:3 * d_model]

    attn = (Q @ K.T)/np.sqrt(d_model)

    exp_attn = np.exp(attn)

    attn_weights = exp_attn/np.sum(exp_attn, axis=-1, keepdims=True)

    attn_output = attn_weights @ V @ W_o

    ffn_output = np.maximum(lnorm_x @ W1, 0) @ W2

    return x + attn_output + ffn_output


