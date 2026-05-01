import numpy as np
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
	# Your code here
	y_true, y_pred = np.array(y_true), np.array(y_pred)
	pairs = [tuple(pair) for pair in zip(y_true, y_pred)]
	counts = Counter(pairs)
	tp = counts.get((1, 1), 0)
	fn = counts.get((1, 0), 0)
	fp = counts.get((0, 1), 0)
	tn = counts.get((0, 0), 0)
	if tp == 0 and fn == 0:
		return 0.0
	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	f1 = 2 * precision * recall / (precision + recall)
	return round(f1,3)