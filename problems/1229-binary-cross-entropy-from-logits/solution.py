import torch

def bce_with_logits(logits, targets):
    """Mean BCE-with-logits loss, numerically stable, rounded to 4 decimals.

    Args:
        logits (torch.Tensor): 1-D raw logits.
        targets (torch.Tensor): 1-D binary targets in {0, 1}, same shape.

    Returns:
        float: mean loss rounded to 4 decimal places.
    """
    # TODO: stable BCE-with-logits, mean, round to 4 decimals
    o1 = torch.max(logits, torch.zeros_like(logits))
    o2 = logits * targets
    o3 = torch.log(1 + torch.exp(-torch.abs(logits)))

    out = o1 - o2 + o3

    out_mean = torch.mean(out)

    return round(float(out_mean), 4)
