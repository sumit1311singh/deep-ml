import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        # TODO: store d_head, create q/k/v/out projections (all bias=False)
        self.d_head = d_model//num_heads
        self.num_heads = num_heads
        self.d_model = d_model

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        # x: (B, T, d_model); mask: (T, T) of 0 and -inf, or None
        # TODO: project, reshape into heads, scaled dot-product, mask, softmax, combine, reshape back, out_proj
        B, T, d_model = x.shape
        
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        Q_heads = Q.reshape(B, T, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        K_heads = K.reshape(B, T, self.num_heads, self.d_head).permute(0, 2, 1, 3)
        V_heads = V.reshape(B, T, self.num_heads, self.d_head).permute(0, 2, 1, 3)

        attention_scores = Q_heads @ K_heads.transpose(-2, -1)
        attention_score = attention_scores/math.sqrt(self.d_head)
        
        if mask is not None:
            attention_score = attention_score + mask

        softmax_attention_score = torch.softmax(attention_score, dim=-1)

        attention_output = (softmax_attention_score @ V_heads).permute(0, 2, 1, 3)
        attention_output = attention_output.contiguous().reshape(B, T, self.d_model)

        output = self.out_proj(attention_output)
        return output


