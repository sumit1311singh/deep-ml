import numpy as np

def focal_loss(y_true, y_pred, gamma=2.0, alpha=None):
	"""
	Compute Focal Loss for multi-class classification.
	
	Args:
		y_true: Ground truth labels as class indices (list or 1D array)
		y_pred: Predicted probabilities (2D array, shape: [n_samples, n_classes])
		gamma: Focusing parameter (default: 2.0)
		alpha: Class weights (optional, list or 1D array of length n_classes)
	
	Returns:
		float: Average focal loss
	"""
	y_true = np.array(y_true)
	y_pred = np.array(y_pred)
	y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)

	# One-hot encode y_true
	n_classes = y_pred.shape[1]
	y_true_onehot = np.eye(n_classes)[y_true]
	
	# If alpha provided, apply class weights
	if alpha is not None:
		alpha = np.array(alpha)
		alpha_factor = alpha[y_true]
	else:
		alpha_factor = 1.0

    # Compute focal loss
	pt = np.sum(y_true_onehot * y_pred, axis=1)  # probability of true class
	loss = -alpha_factor * (1 - pt) ** gamma * np.log(pt)
	
	return float(np.mean(loss))  # return scalar average