import numpy as np

def layer_normalization(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
	"""
	Perform Layer Normalization.
	"""
	F = X.shape[2]

	x_mean = np.mean(X, axis=(2), keepdims=True)
	x_var = np.var(X, axis=(2), keepdims=True)

	x_hat = (X - x_mean)/np.sqrt(x_var + epsilon)

	y = gamma * x_hat + beta

	return y