import numpy as np

def avg_pool_2d(input_matrix: list[list[float]], pool_size: int) -> list[list[float]]:
	"""
	Perform 2D average pooling on the input matrix.
	
	Args:
		input_matrix: 2D input array of shape (H, W)
		pool_size: Size of the square pooling window
		
	Returns:
		2D array after average pooling of shape (H//pool_size, W//pool_size)
	"""
	x = np.array(input_matrix, dtype=float)
	H, W = x.shape
	p = pool_size

	x = x.reshape(H//p, p, W//p, p)
	x = x.mean(axis=(1, 3))
	
	return x.tolist()