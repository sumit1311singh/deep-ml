import numpy as np

def kernel_function(x1, x2):
	return sum(x1[i]*x2[i] for i in range(len(x1)))
