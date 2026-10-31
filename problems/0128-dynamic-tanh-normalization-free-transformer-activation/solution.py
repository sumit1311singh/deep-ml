import torch

def dynamic_tanh(x: torch.Tensor, alpha: float, gamma: torch.Tensor, beta: torch.Tensor) -> list:
    """
    Implement the Dynamic Tanh (DyT) function.
    
    Args:
        x: Input tensor
        alpha: Scaling factor for input
        gamma: Output scaling factor
        beta: Output shift factor
    
    Returns:
        List of transformed values
    """
    return (gamma * torch.tanh(alpha*x) + beta).tolist()