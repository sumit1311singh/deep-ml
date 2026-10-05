import torch

def clip_grad_norm(parameters, max_norm: float) -> float:
    parameters = list(parameters)

    total_norm_sq = 0.0

    for p in parameters:
        if p.grad is not None:
            total_norm_sq += torch.sum(p.grad ** 2)

    total_norm = torch.sqrt(total_norm_sq)

    if total_norm > max_norm:
        scale = max_norm / total_norm

        for p in parameters:
            if p.grad is not None:
                p.grad.mul_(scale)

    return total_norm.item()