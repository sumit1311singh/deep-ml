from tinygrad import Tensor

def flatten_then_reshape(x: Tensor, new_shape) -> Tensor:
    # TODO: flatten x to 1-D, then rearrange into new_shape
    return Tensor.reshape(x, new_shape)

def transpose_last_two(x: Tensor) -> Tensor:
    # TODO: swap the last two dimensions of x
    return Tensor.transpose(x, -2, -1)
