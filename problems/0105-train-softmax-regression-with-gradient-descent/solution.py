import numpy as np

def softmax(logits):
	return np.exp(logits)/np.sum(np.exp(logits), axis=1, keepdims=True)

def train_softmaxreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for Softmax regression, optimizing parameters with Cross Entropy loss.
	"""
	# Your code here
	n, m = X.shape
	X = np.concatenate([np.ones((n, 1)), X], axis=1)
	c = np.max(y) + 1

	y_onehot = np.zeros((n, c))
	for i in range(len(y)):
		y_onehot[i, y[i]] = 1

	W = np.zeros((m+1, c))

	losses = []

	for _ in range(iterations):
		y_hat = X @ W #+ b # (n,m)@(m,c) -> (n,c)+(1,c) -> (n,c)

		prob = softmax(y_hat) # (n,c)

		loss = -np.sum(y_onehot * np.log(prob)) 
		losses.append(loss)

		dl_dw = X.T @ (prob - y_onehot)
		#dl_db = np.sum(prob - y_onehot, axis=0, keepdims=True)

		W-=learning_rate*dl_dw
		#b-=learning_rate*dl_db

	#coefficients = np.concatenate([b.T, W.T], axis=1)

	return W.T, losses
