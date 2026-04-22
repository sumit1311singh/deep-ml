import numpy as np

def cosine_annealing_warm_restarts(eta_max: float, eta_min: float, T_0: int, T_mult: int, total_epochs: int) -> list:
	"""
	Compute learning rates using cosine annealing with warm restarts.
	
	Args:
		eta_max: Maximum learning rate
		eta_min: Minimum learning rate
		T_0: Number of epochs in first cycle
		T_mult: Cycle length multiplier after each restart
		total_epochs: Total number of epochs
	
	Returns:
		List of learning rates for each epoch, rounded to 4 decimal places
	"""
	result = []
	cycle_len=T_0
	T_curr = 0
	cycle_num = 1
	T_curr = 0
	for step in range(total_epochs):
		if (T_curr == cycle_len):
			cycle_len = T_0 * (T_mult**cycle_num)
			cycle_num+=1
			T_curr = 0
		result.append(round(float(eta_min + 0.5*(eta_max-eta_min)*(1+np.cos(np.pi*T_curr/cycle_len))), 4))
		T_curr+=1
	return result