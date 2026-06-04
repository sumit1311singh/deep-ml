import numpy as np

def square_relu(x: np.ndarray) -> dict:
	"""
	Apply the Square ReLU activation function and compute its derivative.
	
	Args:
		x: Input numpy array of any shape
	
	Returns:
		Dictionary with 'output' and 'derivative' as numpy arrays
	"""
	dictt = {}
	mask = x<=0
	x[mask] = 0
	output = x**2
	derivative = 2*x
	dictt['output'] = np.array(np.round(output, 4))
	dictt['derivative'] = np.array(np.round(derivative, 4))
	return dictt