import math
import numpy as np

def mish(x: float) -> float:
	"""
	Compute the Mish activation function.

	Args:
		x (float): Input value

	Returns:
		float: Mish activation value rounded to 4 decimal places
	"""
	softplus = np.log(1+np.exp(x))
	return x*np.tanh(softplus) 