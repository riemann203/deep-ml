import numpy as np

def rmse(y_true, y_pred):
	y_true = np.asarray(y_true, dtype=float)
	y_pred = np.asarray(y_pred, dtype=float)

	if y_true.shape != y_pred.shape:
		raise ValueError("The input vectors must have the same shape.")

	if y_true.size == 0 or y_pred.size == 0:
		raise ValueError("The input vectors cannot be empty.")

	rmse_res = np.sqrt(np.mean((y_true - y_pred) ** 2))

	return round(rmse_res,3)
