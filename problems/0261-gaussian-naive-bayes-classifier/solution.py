import numpy as np

def gaussian_naive_bayes(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
	"""
	Implements Gaussian Naive Bayes classifier.
	
	Args:
		X_train: Training features (shape: N_train x D)
		y_train: Training labels (shape: N_train)
		X_test: Test features (shape: N_test x D)
	
	Returns:
		Predicted class labels for X_test (shape: N_test)
	"""
	classes = np.unique(y_train)
	n=len(X_train)

	mean, var, prior = [], [], []

	for cls in classes:
		X_c = X_train[y_train==cls]

		mean.append(np.mean(X_c, axis=0))
		var.append(np.var(X_c, axis=0))
		prior.append(len(X_c)/n)

	mean, var, prior = np.array(mean), np.array(var), np.array(prior)

	diff = X_test[:, None, :] - mean

	log_likelihood = - 0.5 * np.sum(np.log(2*np.pi*var + diff**2/var), axis=2)

	log_posterior = np.log(prior) + log_likelihood

	predictions = classes[np.argmax(log_posterior, axis=1)]

	return predictions

