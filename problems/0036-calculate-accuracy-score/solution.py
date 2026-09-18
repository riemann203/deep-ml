import numpy as np

def accuracy_score(y_true, y_pred):
	num_correct_predictions = np.sum(
		np.where(y_true == y_pred, 1, 0)
	)
	return num_correct_predictions / len(y_true)