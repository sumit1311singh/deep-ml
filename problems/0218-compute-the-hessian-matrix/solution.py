from typing import Callable
import numpy as np

def compute_hessian(f: Callable[[list[float]], float], point: list[float], h: float = 1e-5) -> list[list[float]]:
	"""
	Compute the Hessian matrix of function f at the given point using finite differences.
	
	Args:
		f: A scalar function that takes a list of floats and returns a float
		point: The point at which to compute the Hessian (list of coordinates)
		h: Step size for finite differences (default: 1e-5)
		
	Returns:
		The Hessian matrix as a list of lists (n x n where n = len(point))
	"""
	x = np.array(point, dtype=float)
	fx = f(x)  # base output
	#m = fx.size          # number of outputs
	n = x.size           # number of inputs

	H = np.zeros((n, n))

	for i in range(n):
		for j in range(n):
			if i==j:
				x_stepPlus, x_stepMinus = x.copy(), x.copy()
				x_stepPlus[i] += h
				x_stepMinus[i] -= h
				fx_stepPlus = f(x_stepPlus)
				fx_stepMinus = f(x_stepMinus)
				H[i, i] = (fx_stepPlus - 2*fx + fx_stepMinus) / h**2
			else:
				x_stepI, x_stepJ, x_stepIJ = x.copy(), x.copy(), x.copy()
				x_stepI[i] += h
				x_stepJ[j] += h
				x_stepIJ[i] += h
				x_stepIJ[j] += h
				fx_stepIJ = f(x_stepIJ)
				fx_stepI = f(x_stepI)
				fx_stepJ = f(x_stepJ)
				H[i, j] = (fx_stepIJ - fx_stepI - fx_stepJ + fx) / h**2
				H[i, j] = (fx_stepIJ - fx_stepI - fx_stepJ + fx) / h**2

	return H.tolist()