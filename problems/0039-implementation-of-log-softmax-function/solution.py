import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	max_val = np.max(scores)
	shifted_scores = scores - max_val
	exp_shifted_scores = np.exp(shifted_scores)
	log_total = np.log(np.sum(exp_shifted_scores))
	return shifted_scores - log_total