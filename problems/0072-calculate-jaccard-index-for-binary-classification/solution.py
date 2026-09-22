import numpy as np

def jaccard_index(y_true, y_pred):
	y_true = np.asarray(y_true, dtype=int)
	y_pred = np.asarray(y_pred, dtype=int)

	numerator = np.sum(y_true * y_pred)
	one_vector = np.ones_like(y_true)
	denominator = np.sum(one_vector - (one_vector - y_true) * (one_vector - y_pred))
	if denominator == 0:
		raise ValueError("Both arrays contain only zeros.")

	result = numerator / denominator

	return round(result, 3)