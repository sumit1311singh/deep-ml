import numpy as np

def GeLU(x: np.ndarray) -> np.ndarray:
	# Your code here
	scores = [np.round(0.5*i*(1+np.tanh((np.sqrt(2/np.pi))*(i+0.044715*(i**3)))), 4) for i in x]
	return scores