import numpy as np 

def ols(features, target): 
	b = np.dot(np.linalg.inv(np.dot(features.T, features)), np.dot(features.T, target))
	return b

def predict(sample, w): 
	return np.dot(sample, w)