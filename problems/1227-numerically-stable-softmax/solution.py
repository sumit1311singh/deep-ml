import torch

def softmax(t, dim):
    """Numerically stable softmax along dim.

    Args:
        t (torch.Tensor): input tensor
        dim (int): dimension along which to apply softmax

    Returns:
        torch.Tensor: tensor of same shape as t; slices along dim sum to 1
    """
    # TODO: subtract max along dim, exp, then normalize
    t = t - t.max(dim=dim, keepdims=True).values

    exp = torch.exp(t)

    probs = exp / exp.sum(dim=dim, keepdims=True)

    return probs

