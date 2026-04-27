
from collections import Counter

def confusion_matrix(data):
	# Implement the function here
	pairs = [tuple(pair) for pair in data]
	counts = Counter(pairs)

	tp = counts.get((1, 1), 0)
	fn = counts.get((1, 0), 0)
	fp = counts.get((0, 1), 0)
	tn = counts.get((0, 0), 0)

	return [
		[tp, fn],
		[fp, tn]
	]