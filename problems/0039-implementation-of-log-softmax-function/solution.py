import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	x=np.array(scores)
	maxx=np.max(x)
	log_sum_exp = np.log(np.sum(np.exp(x - maxx)))
	return x - maxx - log_sum_exp