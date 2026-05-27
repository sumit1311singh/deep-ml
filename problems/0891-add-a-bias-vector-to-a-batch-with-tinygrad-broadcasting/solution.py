from tinygrad import Tensor

def add_bias(x: Tensor, b: Tensor) -> Tensor:
    # TODO: add b to every row of x using broadcasting
    return x+b
