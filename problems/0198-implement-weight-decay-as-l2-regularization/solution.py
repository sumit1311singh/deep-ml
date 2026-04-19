import numpy as np

def apply_weight_decay(parameters: list[list[float]], gradients: list[list[float]], 
                       lr: float, weight_decay: float, apply_to_all: list[bool]) -> list[list[float]]:
	"""
	Apply weight decay (L2 regularization) to parameters.
	
	Args:
		parameters: List of parameter arrays
		gradients: List of gradient arrays
		lr: Learning rate
		weight_decay: Weight decay factor
		apply_to_all: Boolean list indicating which parameter groups get weight decay
	
	Returns:
		Updated parameters
	"""
	

	updated_params = []
	for p, g, apply_decay in zip(parameters, gradients, apply_to_all):
		p = np.array(p, dtype=float)
		g = np.array(g, dtype=float)
		if apply_decay:
			# Gradient descent + weight decay
			new_p = p - lr * g - lr * weight_decay * p
		else:
			# Gradient descent only
			new_p = p - lr * g
		updated_params.append(new_p.tolist())
	return updated_params