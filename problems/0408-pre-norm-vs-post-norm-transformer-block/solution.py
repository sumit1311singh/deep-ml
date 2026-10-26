import numpy as np

def transformer_block(x: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray, b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray, gamma2: np.ndarray, beta2: np.ndarray, mode: str, eps: float = 1e-5) -> np.ndarray:
	"""
	Apply a transformer block with two sublayers using either pre-norm or post-norm.
	
	Args:
		x: Input array of shape (seq_len, d_model)
		W1, b1: Weights and bias for first sublayer
		W2, b2: Weights and bias for second sublayer
		gamma1, beta1: LayerNorm params for first normalization
		gamma2, beta2: LayerNorm params for second normalization
		mode: 'pre_norm' or 'post_norm'
		eps: Epsilon for numerical stability
	
	Returns:
		Output array of shape (seq_len, d_model)
	"""
	if mode == 'pre_norm':
		x_mean = np.mean(x, axis=-1, keepdims=True)
		x_var = np.var(x, axis=-1, keepdims=True)
		norm_x = gamma1 * (x-x_mean)/np.sqrt(x_var+eps) + beta1
		h = x + norm_x @ W1 + b1

		h_mean = np.mean(h, axis=-1, keepdims=True)
		h_var = np.var(h, axis=-1, keepdims=True)
		
		norm_h = gamma2 * (h-h_mean)/np.sqrt(h_var+eps) + beta2
		y = h + norm_h @ W2 + b2

		return y
		
	elif mode == 'post_norm':
		z = x + x @ W1 + b1
		z_mean = np.mean(z, axis=-1, keepdims=True)
		z_var = np.var(z, axis=-1, keepdims=True)

		h = gamma1 * (z-z_mean)/np.sqrt(z_var+eps) + beta1

		z2 = h + h @ W2 + b2
		z2_mean = np.mean(z2, axis=-1, keepdims=True)
		z2_var = np.var(z2, axis=-1, keepdims=True)

		y = gamma2 * (z2 - z2_mean)/np.sqrt(z2_var+eps) + beta2

		return y

	else:
		raise ValueError('ValueError')