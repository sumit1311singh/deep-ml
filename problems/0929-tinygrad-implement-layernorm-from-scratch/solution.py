from tinygrad import Tensor

def layer_norm(x: Tensor, gamma: Tensor, beta: Tensor, eps: float = 1e-5) -> Tensor:
    # mean and variance over the last dimension
    mean = x.mean(axis=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdim=True)

    # normalize
    x_norm = (x - mean) / (var + eps).sqrt()

    # affine transform
    return x_norm * gamma + beta
