import numpy as np

def adamax_optimizer(parameter, grad, m, u, t, learning_rate=0.002,
                     beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Update parameters using the Adamax optimizer.
    Adamax is a variant of Adam based on the infinity norm.
    """
    # Update biased first moment estimate
    m = beta1 * m + (1 - beta1) * grad

    # Update exponentially weighted infinity norm
    u = np.maximum(beta2 * u, np.abs(grad))

    # Bias correction for the first moment
    m_hat = m / (1 - beta1 ** t)

    # Update parameters
    parameter = parameter - learning_rate * m_hat / (u + epsilon)

    return np.round(parameter, 5), np.round(m, 5), np.round(u, 5)
