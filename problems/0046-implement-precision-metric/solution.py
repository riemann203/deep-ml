import numpy as np
def precision(y_true, y_pred):
	indices = (y_pred == True)
	if len(y_true[indices]) == 0:
		return 0.0
	return float(np.sum(y_true[indices]) / len(y_true[indices]))
