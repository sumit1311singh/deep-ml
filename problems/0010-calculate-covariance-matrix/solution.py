import numpy as np
def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
	# Your code here
	mat = np.array(vectors, dtype=float)
	return np.cov(mat)