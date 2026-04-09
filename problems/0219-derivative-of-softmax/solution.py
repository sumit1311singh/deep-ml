import numpy as np

def softmax_derivative(x: list[float]) -> list[list[float]]:
	"""
	Compute the Jacobian matrix of the softmax function.
	
	Args:
		x: Input vector of real numbers
		
	Returns:
		Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
	"""
	x=np.array(x)
	n=len(x)
	x_sum = sum(np.exp(x))
	softmax = [np.exp(i)/x_sum for i in x]
	J = np.zeros((n,n))
	for i in range(n):
		for j in range(n):
			if i==j:
				J[i][i]=softmax[i]*(1-softmax[i])
			else:
				J[i][j]=-1*softmax[i]*softmax[j]
	return J