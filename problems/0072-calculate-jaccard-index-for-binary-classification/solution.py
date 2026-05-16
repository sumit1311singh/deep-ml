
import numpy as np

def jaccard_index(y_true, y_pred):
	y_pred, y_true = np.array(y_pred), np.array(y_true)
	num = np.sum((y_true==1)&(y_pred==1))
	denom = np.sum((y_true==1)|(y_pred==1))
	result = num/denom
	#print(num, denom, result)
	return round(result, 3)
