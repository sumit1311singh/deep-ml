
import numpy as np

def r_squared(y_true, y_pred):
	y_true, y_pred = np.array(y_true), np.array(y_pred)
	y_true_mean = np.mean(y_true)
	return 1 - np.sum((y_true-y_pred)**2)/np.sum((y_true-y_true_mean)**2)