
import numpy as np

def gini_impurity(y: np.ndarray) -> float:
	"""
	Calculate Gini Impurity for a list of class labels.

	:param y: List of class labels
	:return: Gini Impurity rounded to three decimal places
	"""
	classes, counts = np.unique(y, return_counts=True)
	proportions = counts / counts.sum()
	val = 1.0 - np.sum(proportions ** 2)
	return round(val,3)