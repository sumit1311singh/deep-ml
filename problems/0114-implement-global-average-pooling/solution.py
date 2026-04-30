import numpy as np

def global_avg_pool(x: np.ndarray) -> np.ndarray:
	x=np.array(x, dtype=float)
	result = np.mean(x, axis=(0, 1))
	return result