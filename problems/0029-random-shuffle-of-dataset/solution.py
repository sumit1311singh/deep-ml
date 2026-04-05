import numpy as np

def shuffle_data(X, y, seed=None):
	# Your code here
	if seed is not None:
		np.random.seed(seed)
	n=X.shape[0]
	idx = np.arange(n)
	np.random.shuffle(idx)
	return X[idx], y[idx]