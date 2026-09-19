import numpy as np

def compute_qkv(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray):
	"""
	Compute Query (Q), Key (K), and Value (V) matrices.
	"""
	return np.dot(X, W_q), np.dot(X, W_k), np.dot(X, W_v)

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
	exp_shifted_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
	return exp_shifted_x / np.sum(exp_shifted_x, axis=axis, keepdims=True)

def masked_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray) -> np.ndarray:
	"""
	Compute masked self-attention.
	"""
	d_k = Q.shape[1]
	scores = Q @ K.T / np.sqrt(d_k)
	masked_scores = scores + mask
	weights = softmax(masked_scores, axis=-1)
	return weights @ V