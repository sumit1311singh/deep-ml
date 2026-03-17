import numpy as np
def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	mat = np.array(matrix)
	if mode == 'row':
		means = np.mean(mat, axis=1)
	else:
		means = np.mean(mat, axis=0)
	return means