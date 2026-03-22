import numpy as np

def gradient_direction_magnitude(gradient: list) -> dict:
	"""
	Calculate the magnitude and direction of a gradient vector.
	
	Args:
		gradient: A list representing the gradient vector
	
	Returns:
		Dictionary containing:
		- magnitude: The L2 norm of the gradient
		- direction: Unit vector in direction of steepest ascent
		- descent_direction: Unit vector in direction of steepest descent
	"""
	gradArr = np.array(gradient, dtype=np.float64)
	result = {}
	magnitude = np.linalg.norm(gradArr)
	if magnitude == 0.0:
		result["magnitude"] = 0.0
		result["direction"] = [0.0, 0.0]
		return result
	direction = [i/magnitude for i in gradArr]
	descent_direction = [(-1)*i for i in direction]
	result["magnitude"] = magnitude
	result["direction"] = direction
	result["descent_direction"] = descent_direction
	return result
	