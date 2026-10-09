import numpy as np

def residual_block(x: np.ndarray, w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
	# Your code here
	x_og = x

	x_l1 = x @ w1
	x_a1 = np.maximum(x_l1, 0)

	x_l2 = x_l1 @ w2

	x_l3 = x_l2 + x_og
	x_a3 = np.maximum(x_l3, 0)

	return x_a3