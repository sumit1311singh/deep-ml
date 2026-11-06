import numpy as np

def hash_moe_forward(token_ids, embeddings, expert_weights, num_experts: int) -> np.ndarray:
    """
    Hash-based routing forward pass for an MoE layer.

    Args:
        token_ids: 1D array-like of N integer token IDs
        embeddings: 2D array-like of shape (N, d) of token embeddings
        expert_weights: list of E numpy arrays, each of shape (d, d_out)
        num_experts: number of experts E

    Returns:
        numpy array of shape (N, d_out) with per-token expert outputs.
    """
    token_ids = np.asarray(token_ids)
    embeddings = np.asarray(embeddings)

    # Determine expert assignment using token IDs.
    expert_indices = token_ids % num_experts

    # Allocate output array.
    d_out = expert_weights[0].shape[1]
    outputs = np.empty((len(token_ids), d_out), dtype=np.result_type(embeddings, *expert_weights))

    # Apply the corresponding expert to each token.
    for expert_idx in range(num_experts):
        mask = expert_indices == expert_idx
        if np.any(mask):
            outputs[mask] = embeddings[mask] @ expert_weights[expert_idx]

    return outputs
