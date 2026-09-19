import math
import numpy as np
def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	# Your code here
	features=np.array(features)
	weights=np.array(weights)
	probabilities=1/(1+np.exp(-((features @ weights)+bias)))
	mse=np.mean((probabilities-labels)**2)
	return probabilities, mse