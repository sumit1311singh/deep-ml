import numpy as np
def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
	B = np.array(B, dtype=float)
	C = np.array(C, dtype=float)
	P = np.linalg.inv(C) @ B   # matrix multiplication
	return P.tolist()