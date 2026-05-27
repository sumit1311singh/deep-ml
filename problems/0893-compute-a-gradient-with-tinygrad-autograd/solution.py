from tinygrad import Tensor

def grad_of_quadratic(x_value: float) -> float:
    # TODO: build a tracked tensor for x, compute f(x), run backprop, return df/dx as a float
    x = Tensor(x_value, requires_grad=True)
    f = x**2 + 3*x + 2
    f.backward()
    return x.grad.item()