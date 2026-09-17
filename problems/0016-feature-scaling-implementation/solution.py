import numpy as np
def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	# Your code here
	miu=np.mean(data, axis=0)
	biao=np.std(data, axis=0)
	jida=np.max(data, axis=0)
	jixiao=np.min(data, axis=0)
	standardized_data=[]
	normalized_data=[]
	for i in range(len(data)):
		temp=[]
		temp2=[]
		for j in range(len(data[0])):
			temp.append((data[i][j]-miu[j])/biao[j])
			temp2.append((data[i][j]-jixiao[j])/(jida[j]-jixiao[j]))
		standardized_data.append(temp)
		normalized_data.append(temp2)
	
	return standardized_data, normalized_data