import numpy as np
def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
	A = np.array(A, dtype=float)
	b = np.array(b, dtype=float)
	m = len(b)

    # Initialize solution vector with zeros
	x = np.zeros(m)

	for _ in range(n):
		x_new = np.zeros_like(x)
		for i in range(m):
            # sum of a_ij * x_j for j != i
			s = sum(A[i, j] * x[j] for j in range(m) if j != i)
			x_new[i] = (b[i] - s) / A[i, i]
        # round each intermediate solution to 4 decimals
		x = np.round(x_new, 4)
	
	return x.tolist()