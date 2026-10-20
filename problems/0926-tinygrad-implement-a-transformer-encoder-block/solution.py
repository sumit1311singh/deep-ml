from tinygrad import Tensor, nn

class TransformerBlock:
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        # TODO: norms, projections, MLP, store dims and dropout prob
        self.d_model = d_model
        self.num_heads = num_heads

        self.d_head = d_model//num_heads

        self.norm1 = nn.LayerNorm(d_model)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_out = nn.Linear(d_model, d_model, bias=False)

        self.norm2 = nn.LayerNorm(d_model)

        self.mlp_fc1 = nn.Linear(d_model, d_ff)
        self.mlp_fc2 = nn.Linear(d_ff, d_model)

        self.dropout_p = dropout

    def __call__(self, x):
        # TODO: pre-LN attn sublayer, then pre-LN MLP sublayer, both with residual + dropout

        normalized_x = self.norm1(x)

        q = self.q_proj(normalized_x)
        k = self.k_proj(normalized_x)
        v = self.v_proj(normalized_x)

        B, T, _ = x.shape

        q = q.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)
        k = k.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)
        v = v.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)

        attn = q.scaled_dot_product_attention(k, v)
        attn = attn.transpose(1, 2).reshape(B, T, self.d_model)

        attn_out = self.attn_out(attn)

        x2 = x + attn_out.dropout(self.dropout_p)

        normalized_x2 = self.norm2(x2)

        fc1_out = self.mlp_fc1(normalized_x2).gelu()
        fc2_out = self.mlp_fc2(fc1_out)
        
        output = x + fc2_out.dropout(self.dropout_p)

        return output




        
