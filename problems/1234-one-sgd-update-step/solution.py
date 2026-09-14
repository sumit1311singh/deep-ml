import torch

def sgd_step(w, grad, lr):
    """Perform one SGD update step.

    Args:
        w: Current parameter tensor.
        grad: Gradient tensor (same shape as w).
        lr: Learning rate (float).

    Returns:
        Updated parameter tensor w - lr * grad.
    """
    # TODO
    w_new = w - lr * grad
    return w_new