import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    """
    Perform numerical gradient checking using centered finite differences.
    
    Args:
        f: A function that takes a numpy array and returns a scalar
        x: numpy array, the point at which to check gradient
        analytical_grad: numpy array, the analytically computed gradient
        epsilon: float, small value for finite difference approximation
    
    Returns:
        tuple: (numerical_grad, relative_error)
    """
    x = np.array(x, dtype=float)
    numerical_grad = np.zeros_like(x)

    # Compute numerical gradient coordinate by coordinate
    for i in range(len(x)):
        perturb = np.zeros_like(x)
        perturb[i] = epsilon
        fx_plus = f(x + perturb)
        fx_minus = f(x - perturb)
        numerical_grad[i] = (fx_plus - fx_minus) / (2 * epsilon)

    # Relative error
    num = np.linalg.norm(analytical_grad - numerical_grad)
    denom = np.linalg.norm(analytical_grad) + np.linalg.norm(numerical_grad)
    error = num / denom if denom != 0 else 0.0

    return numerical_grad, error
    