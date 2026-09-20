
import numpy as np

def r_squared(y_true, y_pred):
	y_true = np.asarray(y_true)
	y_pred = np.asarray(y_pred)

	if len(y_true) == 0:
		return np.nan

	sst = np.sum((y_true - np.mean(y_true)) ** 2)
	ssr = np.sum((y_true - y_pred) ** 2)
	if sst == 0.0:
		return 1.0 if ssr == 0.0 else -np.inf
	return 1.0 - ssr / sst