from collections import Counter

def calculate_f1_score(y_true, y_pred):
	"""
	Calculate the F1 score based on true and predicted labels.

	Args:
		y_true (list): True labels (ground truth).
		y_pred (list): Predicted labels.

	Returns:
		float: The F1 score rounded to three decimal places.
	"""
	if len(y_true) != len(y_pred):
		raise ValueError("Predicted and true labels must have the same length.")

	counts = Counter(zip(y_true, y_pred))
	tp = counts[(1, 1)]
	fp = counts[(0, 1)]
	fn = counts[(1, 0)]
	if tp == fp == fn == 0:
		return 0.0

	f1 = 2 * tp / (2 * tp + fp + fn)
	return round(f1,3)