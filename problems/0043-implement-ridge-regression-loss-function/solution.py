import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
	y_hat = np.dot(X, w)
	loss = np.mean((y_hat - y_true)**2)
	penalty = alpha * np.sum(np.square(w))
	return loss + penalty

