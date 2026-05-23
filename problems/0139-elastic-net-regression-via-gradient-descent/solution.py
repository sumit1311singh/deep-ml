import numpy as np

def elastic_net_gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha1: float = 0.1,
    alpha2: float = 0.1,
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    tol: float = 1e-4,
) -> tuple:
    # Implement Elastic Net regression here
    m, n = X.shape
    w, b = np.zeros(n), 0

    for i in range(max_iter):
        y_hat = np.dot(X, w) + b

        error = y_hat - y

        # mse = 1/(2*m)*np.sum((error)**2)
        # l1 = alpha1*np.sum(np.linalg.norm(w))
        # l2 = alpha2*np.sum(np.linalg.norm(w**2))

        w_g = (1/m)*np.dot(X.T, error) + alpha1*np.sign(w) + 2*alpha2*w
        b_g = np.mean(error)

        w_old = w.copy()
        w-=learning_rate*w_g
        b-=learning_rate*b_g

        if(np.sum(np.abs(w_g)) < tol):
            break
        
    return w, b