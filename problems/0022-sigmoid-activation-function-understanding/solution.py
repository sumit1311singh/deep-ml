import math

def sigmoid(z: float) -> float:
	result = 1 / (1 + math.exp(-1*z))
	return result