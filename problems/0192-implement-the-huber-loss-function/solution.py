import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
	"""
	Compute the Huber Loss between true and predicted values.

	Args:
		y_true (float | list[float]): Ground truth values
		y_pred (float | list[float]): Predicted values
		delta (float): Transition threshold between MSE and MAE behavior

	Returns:
		float: Average Huber loss
	"""
	y_true, y_pred = np.array(y_true), np.array(y_pred)
	result = np.mean(np.where(abs(y_true-y_pred)<=delta, 
					(1/2)*((y_true-y_pred)**2),
					delta*(abs(y_true-y_pred) - (1/2)*delta)))
	return round(result, 4)