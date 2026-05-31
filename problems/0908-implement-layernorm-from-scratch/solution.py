import torch

def layer_norm(x, gamma, beta, eps=1e-5):
    # TODO: normalize over the last dim, then affine-transform with gamma and beta
    x_mean = x.mean(dim=-1, keepdim=True)
    x_std = x.std(dim=-1, unbiased=False, keepdim=True)
    l_norm = gamma * (x - x_mean)/torch.sqrt(x_std**2 + eps) + beta
    return l_norm
