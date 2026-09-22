
import numpy as np

def dice_score(y_true, y_pred):
	y_true = np.asarray(y_true, dtype=int)
	y_pred = np.asarray(y_pred, dtype=int)

	if y_true.ndim != 1 or y_true.shape != y_pred.shape:
		raise ValueError("Inputs must be 1D vectors of equal length.")

	if not (
		np.all(np.isin(y_true, [0, 1])) and np.all(np.isin(y_pred, [0, 1]))
	):
		raise ValueError("Inputs must contain only 0 and 1.")

	intersection = np.count_nonzero(y_true & y_pred)
	cardinal_y_pred = np.count_nonzero(y_pred)
	cardinal_y_true = np.count_nonzero(y_true)

	if cardinal_y_pred + cardinal_y_true == 0.0:
		return 0.0

	res = 2 * intersection / (cardinal_y_pred + cardinal_y_true)

	return round(res, 3)
