import numpy as np

def f_score(y_true, y_pred, beta):
	"""
	Calculate F-Score for a binary classification task.

	:param y_true: Numpy array of true labels
	:param y_pred: Numpy array of predicted labels
	:param beta: The weight of precision in the harmonic mean
	:return: F-Score rounded to three decimal places
	"""
	positive_mask = (y_true == 1.0)
	recall = np.sum(y_pred[positive_mask]) / len(y_pred[positive_mask])

	positive_mask = (y_pred == 1.0)
	precision = np.sum(y_true[positive_mask]) / len(y_true[positive_mask])

	return round(
		1.0 / (((beta ** 2 / recall) + 1.0 / precision) / (1 + beta ** 2))
	, 3)

