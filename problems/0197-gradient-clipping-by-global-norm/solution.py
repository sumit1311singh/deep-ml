import numpy as np

def clip_gradients_by_global_norm(gradients: list[list[float]], max_norm: float) -> list[list[float]]:
	"""
	Clip gradients by global norm.
	
	Args:
		gradients: List of gradient arrays
		max_norm: Maximum allowed global norm
	
	Returns:
		List of clipped gradient arrays
	"""
	gradients = [np.array(g) for g in gradients]
	global_norm = np.sqrt(sum(sum(g**2) for g in gradients))

	if global_norm <= max_norm:
		return [g.tolist() for g in gradients]

	scale = max_norm / global_norm
	clipped_gradients = [
        (g * scale).tolist()
        for g in gradients
    ]
	return clipped_gradients