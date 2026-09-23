import numpy as np

def sigmoid(z):
	return 1/(1+np.exp(-z))
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):
	# Your code here
	mse_values=[]
	n = len(labels)
	for i in range(epochs):
		# 计算mse
		# 原本的
		y_mao=sigmoid(features @ initial_weights+initial_bias)
		mse=np.mean((y_mao-labels)**2)
		mse_values.append(np.round(mse,4))
		# 再进行权重更新
		initial_weights=initial_weights-(2/n)*learning_rate* features.T @ ((y_mao-labels) * y_mao * (1-y_mao))
		# 偏执更新
		initial_bias=initial_bias-(2/n)*learning_rate*np.sum((y_mao-labels) * y_mao * (1-y_mao))
		

		

	updated_weights=initial_weights
	updated_bias=initial_bias
	
	return np.round(updated_weights,4).tolist(),np.round(updated_bias,4),mse_values