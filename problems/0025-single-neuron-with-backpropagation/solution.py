import numpy as np
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	mse_values = []
	features = np.array(features, dtype=float)
	labels = np.array(labels, dtype=float)
	initial_weights = np.array(initial_weights, dtype=float)
	initial_bias = float(initial_bias)
	m, n = features.shape
	for _ in range(epochs):
		# Forward pass
		z = np.dot(features, initial_weights) + initial_bias
		y_hat = 1/(1+np.exp(-1*z))

		# Loss
		loss = np.mean((y_hat-labels)**2)
		mse_values.append(np.round(loss, 4))

		# Backprop (Gradients)
 		# find: dLoss_dW
		dLoss_dyHat = (2/m)*(y_hat-labels)
		dyHat_dz = y_hat*(1-y_hat)
		dz_dW = features
		#dz_dB = 1

		# Chain rule
		dL_dz = dLoss_dyHat * dyHat_dz
		dW = np.dot(dz_dW.T, dL_dz) # dz_dW = features
		db = np.sum(dL_dz) # dL_dz * dz_dB where dz_dB = 1 

		initial_weights -= learning_rate * dW
		initial_bias -= learning_rate * db

		#initial_weights = np.round(initial_weights, 4)
		#initial_bias = np.round(initial_bias, 4)

	return np.round(initial_weights, 4), np.round(initial_bias, 4), mse_values