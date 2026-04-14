import numpy as np

def entropy_and_cross_entropy(P: list[float], Q: list[float]) -> tuple[float, float]:
	"""
	Compute entropy of P and cross-entropy between P and Q.
	
	Args:
		P: True probability distribution
		Q: Predicted probability distribution
	
	Returns:
		Tuple of (entropy H(P), cross-entropy H(P,Q))
	"""
	p, q=np.array(P), np.array(Q)
	eps = 1e-15 
	p, q=np.clip(p, eps, 1.0), np.clip(q, eps, 1.0)
	entropy = -1*sum(p*np.log(p))
	crossEntropy = -1*sum(p*np.log(q))
	return entropy, crossEntropy