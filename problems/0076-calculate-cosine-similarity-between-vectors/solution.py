import numpy as np

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
	"""
	Calculate the cosine_similarity of two vectors.
	Args:
		vec1 (numpy.ndarray): 1D array representing the first vector.
		vec2 (numpy.ndarray): 1D array representing the second vector.
	Returns:
		The cosine_similarity of the two vectors.
	"""
	v1 = np.asarray(v1, dtype=float)
	v2 = np.asarray(v2, dtype=float)

	if (v1.ndim != 1 or v2.ndim != 1) or (v1.shape != v2.shape):
		raise ValueError("v1 and v2 must be 1D vectors of equal length.")

	norm_v1 = np.linalg.norm(v1)
	norm_v2 = np.linalg.norm(v2)
	denominator = norm_v1 * norm_v2
	if denominator == 0.0:
		raise ValueError("Cosine similarity is not defined for zero vectors.")

	return float(np.dot(v1, v2) / denominator)
