from tinygrad import Tensor

def linear_forward(x: Tensor, W: Tensor, b: Tensor) -> Tensor:
    # TODO: implement y = x W^T + b using tinygrad ops
    return Tensor.matmul(x, W.T) + b
