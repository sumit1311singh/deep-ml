import numpy as np


def cross_attention(decoder_x, encoder_out, W_q, W_k, W_v, src_pad_mask=None):
    """
    Encoder-decoder attention: queries from decoder states, keys and values from encoder output.

    decoder_x:    (tgt_len, d_model)
    encoder_out:  (src_len, d_model)
    W_q, W_k:     (d_model, d_k)
    W_v:          (d_model, d_v)
    src_pad_mask: optional (src_len,) boolean / 0-1 array, True or 1 at padded source positions

    Returns (output, weights) with shapes (tgt_len, d_v) and (tgt_len, src_len).
    Masked source positions must have weight exactly 0.
    """
    Q = decoder_x @ W_q # (tgt_len, d_k)
    K = encoder_out @ W_k # (src_len, d_k)
    V = encoder_out @ W_v # (src_len, d_k)

    d_k = K.shape[1]

    cross_attention_score = (Q @ K.T)/np.sqrt(d_k) # (tgt_len, src_len)

    if src_pad_mask is not None:
        cross_attention_score = cross_attention_score + np.where(src_pad_mask==True, -np.inf, 0)

    shifted_cross_attention_score = cross_attention_score - np.max(cross_attention_score, axis=-1, keepdims=True)

    exp_cross_attention_score = np.exp(shifted_cross_attention_score)

    softmax_cross_attention_score = exp_cross_attention_score/np.sum(exp_cross_attention_score, axis=-1, keepdims=True)

    return softmax_cross_attention_score @ V, softmax_cross_attention_score # (tgt_len, d_k), (tgt_len, src_len)


