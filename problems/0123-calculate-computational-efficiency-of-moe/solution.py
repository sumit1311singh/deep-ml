def compute_efficiency(n_experts, k_active, d_in, d_out):
    """
    Calculate computational savings of MoE vs. dense layer.

    Args:
        n_experts: Total number of experts
        k_active: Number of active experts (sparsity)
        d_in: Input dimension
        d_out: Output dimension

    Returns:
        Percentage savings in FLOPs
    """
    dense_layer_cost = n_experts * d_in * d_out
    moe_cost = k_active * d_in * d_out

    result = 100 * (dense_layer_cost - moe_cost)/dense_layer_cost

    return result