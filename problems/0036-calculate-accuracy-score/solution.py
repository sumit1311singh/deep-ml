import numpy as np

def accuracy_score(y_true, y_pred):
	cnt = 0
	n=len(y_true)
	for i in range(n):
		if y_true[i]==y_pred[i]:
			cnt+=1
	return cnt/n