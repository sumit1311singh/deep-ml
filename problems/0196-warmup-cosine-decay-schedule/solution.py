import numpy as np

def warmup_cosine_schedule(T: int, W: int, lr_max: float, lr_min: float) -> list[float]:
	"""
	Compute learning rate schedule with linear warmup and cosine decay.
	
	Args:
		T: Total number of training steps
		W: Number of warmup steps
		lr_max: Maximum learning rate (reached after warmup)
		lr_min: Minimum learning rate (reached at end of training)
	
	Returns:
		List of learning rates for each step
	"""
	result = []
	for step in range(T):
		if step<W:
			result.append(round(float(lr_max*step/W), 4))
		else:
			result.append(round(float(lr_min + 0.5*(lr_max-lr_min)*(1+np.cos((step-W)*np.pi/(T-W)))), 4))
	return result