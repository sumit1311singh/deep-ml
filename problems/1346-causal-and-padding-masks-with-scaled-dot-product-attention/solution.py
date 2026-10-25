import torch
import torch.nn as nn
import torch.nn.functional as F

def attention(q, k, v, key_padding_mask=None, causal=False):
    # TODO: build a single bool attn_mask broadcastable to (B, H, T_q, T_k)
    #       where True means "may attend"
    # TODO: combine the causal lower-triangular mask and the key padding mask
    # TODO: return F.scaled_dot_product_attention(q, k, v, attn_mask=...)
    attn_mask = None

    if key_padding_mask is not None:
        # (B, T_k) -> (B, 1, 1, T_k)
        attn_mask = key_padding_mask[:, None, None, :]

    if causal:
        T_q, T_k = q.shape[-2], k.shape[-2]

        # (T, T), True means attention is allowed
        causal_mask = torch.ones(
            T_q, T_k, dtype=torch.bool, device=q.device
        ).tril()

        if attn_mask is None:
            attn_mask = causal_mask
        else:
            attn_mask = attn_mask & causal_mask

    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask
    )
