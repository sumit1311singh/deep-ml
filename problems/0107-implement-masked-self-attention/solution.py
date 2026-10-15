import numpy as np

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
	"""
	Compute Query (Q), Key (K), and Value (V) matrices.
	"""
	return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
	"""
	Compute masked self-attention.
	"""
	# Your code here
	d_k = K.shape[1]
	attention_score = (Q @ K.T)/np.sqrt(d_k)

	masked_attention_score = attention_score + mask
	shifted_masked_attention_score = masked_attention_score - np.max(masked_attention_score, axis=1, keepdims=True)

	exp_attention_scores = np.exp(shifted_masked_attention_score)
	softmax_masked_attention_score = exp_attention_scores/np.sum(exp_attention_scores, axis=1, keepdims=True)

	return softmax_masked_attention_score @ V