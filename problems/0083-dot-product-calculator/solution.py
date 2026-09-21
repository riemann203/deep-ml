import numpy as np

def calculate_dot_product(vec1: np.ndarray, vec2: np.ndarray) -> float:
	"""
	Calculate the dot product of two vectors.
	Args:
		vec1 (numpy.ndarray): 1D array representing the first vector.
		vec2 (numpy.ndarray): 1D array representing the second vector.
	Returns:
		The dot product of the two vectors.
	"""
	vec1 = np.asarray(vec1, dtype=float)
	vec2 = np.asarray(vec2, dtype=float)
	return float(np.dot(vec1, vec2))