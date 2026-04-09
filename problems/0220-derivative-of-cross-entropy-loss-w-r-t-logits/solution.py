import numpy as np

def cross_entropy_derivative(logits: list[float], target: int) -> list[float]:
	"""
	Compute the derivative of cross-entropy loss with respect to logits.
	
	Args:
		logits: Raw model outputs (before softmax)
		target: Index of the true class (0-indexed)
		
	Returns:
		Gradient vector where gradient[i] = dL/d(logits[i])
	"""
	logits=np.array(logits)
	n=len(logits)
	y=np.zeros(n)
	y[target] = 1
	logitsSum = sum(np.exp(logits))
	softmax = [np.exp(i)/logitsSum for i in logits]
	return (softmax-y).tolist()