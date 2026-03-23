import numpy as np
import math

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	# Your code here
	features, labels, weights = np.array(features, dtype=float), np.array(labels, dtype=float), np.array(weights, dtype=float)
	newLabels = np.dot(features, weights) + bias
	probabilities = 1/(1+np.exp(-1*newLabels))
	mse = np.round(np.mean((probabilities-labels)**2), 4)
	probabilities = np.round(probabilities, 4).tolist()
	return probabilities, mse