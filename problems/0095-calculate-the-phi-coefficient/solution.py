import numpy as np


def phi_corr(x: list[int], y: list[int]) -> float:
	"""
	Calculate the Phi coefficient between two binary variables.

	Args:
	x (list[int]): A list of binary values (0 or 1).
	y (list[int]): A list of binary values (0 or 1).

	Returns:
	float: The Phi coefficient rounded to 4 decimal places.
	"""
	x = np.asarray(x, dtype=float)
	y = np.asarray(y, dtype=float)
	val = np.corrcoef(x, y)[0, 1]
	if np.isnan(val):
		return 0.0

	return round(float(val), 4)