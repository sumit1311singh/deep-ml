import numpy as np 
def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	x = np.array(data, dtype=np.float64)

	mean = np.mean(x, axis = 0, keepdims = True)
	std = np.std(x, axis = 0, keepdims = True)
	standardized_data = np.round((x-mean)/std, 4).tolist()

	minn = np.min(x, axis = 0, keepdims = True)
	maxx = np.max(x, axis = 0, keepdims = True)
	normalized_data = np.round((x-minn)/(maxx-minn), 4).tolist()

	return standardized_data, normalized_data