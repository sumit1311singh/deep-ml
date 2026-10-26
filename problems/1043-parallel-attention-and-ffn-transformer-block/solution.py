import numpy as np

def parallel_block(x, gamma, beta, Wq, Wk, Wv, Wo, W1, b1, W2, b2):
    """
    Parallel attention + FFN transformer block.
    Returns a list of lists of shape (T, D).
    """
    x_mean = np.mean(x, axis=-1, keepdims=True)
    x_var = np.var(x, axis=-1, keepdims=True)
    eps = 1e-5

    lnorm_x = gamma * (x-x_mean)/np.sqrt(x_var+eps) + beta

    Q, K, V = lnorm_x @ Wq, lnorm_x @ Wk, lnorm_x @ Wv
    
    d_model = K.shape[1]
    attn = (Q @ K.T)/np.sqrt(d_model)

    exp_attn = np.exp(attn)

    attn_weights = exp_attn/np.sum(exp_attn, axis=-1, keepdims=True)

    attn_output = attn_weights @ V @ Wo

    ffn_output = np.maximum(0, lnorm_x @ W1 + b1) @ W2 + b2

    y = x + attn_output + ffn_output

    return y
