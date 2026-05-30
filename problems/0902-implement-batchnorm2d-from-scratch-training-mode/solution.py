import torch

def batchnorm2d(x, gamma, beta, eps=1e-5):
    # TODO: training-mode batchnorm2d

    x_mean = x.mean(dim=(0, 2, 3), keepdim=True)
    x_var = x.var(dim=(0, 2, 3), unbiased=False, keepdim=True)

    x_norm = (x - x_mean) / torch.sqrt(x_var + eps)
    
    gamma = gamma.reshape(1, x_mean.shape[1], 1, 1)
    beta = beta.reshape(1, x_mean.shape[1], 1, 1)
    
    return x_norm * gamma + beta
