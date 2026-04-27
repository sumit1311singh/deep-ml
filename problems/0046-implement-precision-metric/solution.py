from collections import Counter

def precision(y_true, y_pred):
	pairs = [tuple((i, j)) for i, j in zip(y_true, y_pred)]
	counts = Counter(pairs)

	tp = counts.get((1, 1), 0)
	fp = counts.get((0, 1), 0)

	return tp/(tp+fp)
