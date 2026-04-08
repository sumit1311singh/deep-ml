import numpy as np

def activation_derivatives(x: float) -> dict[str, float]:
	"""
	Compute the derivatives of Sigmoid, Tanh, and ReLU at a given point x.
	
	Args:
		x: Input value
		
	Returns:
		Dictionary with keys 'sigmoid', 'tanh', 'relu' and their derivative values
	"""
	sigmoid = 1/(1+np.exp(x))
	tanh = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
	dictt = {}
	dictt['sigmoid'] = np.round(sigmoid*(1-sigmoid), 4)
	dictt['tanh'] = np.round((1-tanh**2), 4)
	dictt['relu'] = 1.0 if x > 0 else 0.0
	return dictt