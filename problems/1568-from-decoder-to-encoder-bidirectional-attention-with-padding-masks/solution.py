import numpy as np


def causal_mask(T):
    # TODO: (T, T) boolean, True where j <= i
    mask = np.tril(np.ones((T, T), dtype=bool))
    return mask


def bidirectional_mask(pad):
    # TODO: (B, T, T) boolean, True where key j is a real token
    real_keys = (pad == 1)
    real_keys = real_keys[:, None, :]
    mask = np.broadcast_to(real_keys, (pad.shape[0], pad.shape[1], pad.shape[1]))
    return mask


def masked_attention(Q, K, V, allowed):
    # TODO: softmax(QK^T / sqrt(d)) with disallowed pairs at -inf, times V, rounded to 4 decimals
    _, _, d_k = K.shape

    attention_score = (Q @ K.transpose(0, 2, 1)) / np.sqrt(d_k)

    attention_score = np.where(allowed, attention_score, -np.inf)

    attention_score = attention_score - np.max(attention_score, axis=-1, keepdims=True)

    exp_attention_score = np.exp(attention_score)
    attention_weights = exp_attention_score/np.sum(exp_attention_score, axis=-1, keepdims=True)

    output = attention_weights @ V

    return np.round(output, 4)


def allowed_pairs(allowed):
    # TODO: number of True entries as an int
    return int(np.count_nonzero(allowed))
