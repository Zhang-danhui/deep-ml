import numpy as np

def log_softmax(scores: list) -> np.ndarray:
	scores=np.array(scores)
	scores=scores-np.max(scores)
	fenzi=np.exp(scores)
	
	fenmu=np.sum(fenzi, axis=0)
	return np.log(fenzi/fenmu)