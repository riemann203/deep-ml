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
	P = np.array(P)
	Q = np.array(Q)
	H_P = -np.sum(np.where(P > 0.0, P * np.log(P), 0.0))
	H_P_Q = -np.sum(P * np.log(Q))
	return H_P, H_P_Q