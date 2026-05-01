import numpy as np
from collections import Counter

def f_score(y_true, y_pred, beta):
	"""
	Calculate F-Score for a binary classification task.

	:param y_true: Numpy array of true labels
	:param y_pred: Numpy array of predicted labels
	:param beta: The weight of precision in the harmonic mean
	:return: F-Score rounded to three decimal places
	"""
	y_true, y_pred = np.array(y_true), np.array(y_pred)
	pairs = [tuple(pair) for pair in zip(y_true, y_pred)]
	counts = Counter(pairs)
	tp = counts.get((1, 1), 0)
	fn = counts.get((1, 0), 0)
	fp = counts.get((0, 1), 0)
	tn = counts.get((0, 0), 0)
	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	denom = precision * beta**2 + recall
	if denom == 0:
		return 0.0
	f_score = (1 + beta**2)*(precision * recall)/denom
	return round(f_score, 3)

