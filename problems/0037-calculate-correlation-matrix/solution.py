import numpy as np

def calculate_correlation_matrix(X, Y=None):
	if Y is None:
		Y=X

	X = np.array(X)
	Y = np.array(Y)

	Xc = X - X.mean(axis=0)
	Yc = Y - Y.mean(axis=0)

	cov = Xc.T @ Yc / (X.shape[0] - 1)

	sx = Xc.std(axis=0, ddof=1)
	sy = Yc.std(axis=0, ddof=1)

	denom = np.outer(sx, sy)

	return cov/denom

		