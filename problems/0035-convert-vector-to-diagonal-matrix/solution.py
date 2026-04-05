import numpy as np

def make_diagonal(x):
	n=len(x)
	mat = np.zeros((n, n))
	for i in range(n):
		mat[i][i]=x[i]
	return mat