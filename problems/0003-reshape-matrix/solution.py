import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
	mat = np.array(a)
	if mat.size != new_shape[0]*new_shape[1]:
		return []
	reshaped_matrix = mat.reshape(new_shape[0], new_shape[1])
	return reshaped_matrix