import torch

def flash_attention_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                            block_size: int = 2) -> torch.Tensor:
    """
    Compute attention output using Flash Attention v1 algorithm.
    
    Args:
        Q: Query matrix (seq_len, d_model)
        K: Key matrix (seq_len, d_model)
        V: Value matrix (seq_len, d_model)
        block_size: Size of blocks for tiled computation
    
    Returns:
        Output matrix (seq_len, d_model)
    """
    N, d = Q.shape
    scale = d ** -0.5
    O = torch.zeros_like(Q)

    for i in range(0, N, block_size):
        q = Q[i:i + block_size]
        m = torch.full((q.shape[0], 1), float("-inf"), dtype=Q.dtype, device=Q.device)
        l = torch.zeros((q.shape[0], 1), dtype=Q.dtype, device=Q.device)
        acc = torch.zeros((q.shape[0], V.shape[1]), dtype=V.dtype, device=V.device)

        for j in range(0, N, block_size):
            k = K[j:j + block_size]
            v = V[j:j + block_size]

            scores = (q @ k.T) * scale
            block_max = scores.max(dim=-1, keepdim=True).values
            new_m = torch.maximum(m, block_max)

            alpha = torch.exp(m - new_m)
            p = torch.exp(scores - new_m)

            l = alpha * l + p.sum(dim=-1, keepdim=True)
            acc = alpha * acc + p @ v
            m = new_m

        O[i:i + q.shape[0]] = acc / l

    return O