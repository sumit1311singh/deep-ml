import numpy as np

def optimizer_step(param, grad, state, lr):
    '''
    Update a parameter using its gradient.
    
    Args:
        param: numpy array - current parameter values (any shape)
        grad: numpy array - gradient of loss w.r.t. param (same shape)
        state: dict - persists between calls, use to store any needed values
                      Example: state = {'step': 5, 'momentum': np.array([...])}
                      First call: state = {} (empty dict)
        lr: float - learning rate
    
    Returns:
        new_param: numpy array - updated parameter (must be same shape as param)
        state: dict - updated state dictionary
    '''
    # TODO: Implement your optimizer update rule
    # Hint: Think about gradient descent and how to use the gradient to update the parameter
    m = state.get('m', 0)
    v = state.get('v', 0)
    t = state.get('t', 1)

    eps, lam = 1e-9, 0.1
    b1, b2 = 0.9, 0.999

    m_new = m*b1 + (1-b1)*grad
    v_new = v*b2 + (1-b2)*grad**2

    m_hat = m_new/(1-b1**t)
    v_hat = v_new/(1-b2**t)

    if grad is 0:
        param = param -  lr * lam * param

    new_param = param - lr * m_hat/(np.sqrt(v_hat) + eps)

    state['m'] = m_new
    state['v'] = v_new
    state['t'] = t+1
    
    return new_param, state
