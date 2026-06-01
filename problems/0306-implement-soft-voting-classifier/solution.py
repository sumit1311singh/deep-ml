import numpy as np

def soft_voting_classifier(probabilities: np.ndarray, weights: list = None) -> list:
	"""
	Implement soft voting for ensemble classification.
	
	Args:
		probabilities: 3D array of shape (n_classifiers, n_samples, n_classes)
		weights: Optional list of weights for each classifier
	
	Returns:
		List of predicted class labels for each sample
	"""
	if weights!=None:
		weights=np.array(weights)
	else:
		weights = np.ones(probabilities.shape[0]) / probabilities.shape[0]
	weighted_avg = np.average(probabilities, axis=0, weights=weights)
	final_preds = np.argmax(weighted_avg, axis=1)
	return final_preds