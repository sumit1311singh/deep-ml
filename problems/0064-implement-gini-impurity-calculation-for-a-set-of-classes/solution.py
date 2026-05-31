
import numpy as np

def gini_impurity(y):
	"""
	Calculate Gini Impurity for a list of class labels.

	:param y: List of class labels
	:return: Gini Impurity rounded to three decimal places
	"""
	y=np.array(y)
	if len(y)==0:
		return 0.0
	_, counts = np.unique(y, return_counts=True)
	probs = counts/counts.sum()
	val = 1-sum(probs**2)
	return round(val,3)