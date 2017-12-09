import numpy as np

def sigmoid(x):
	return 1/(1+e^(-1*x))

def sigmoid_prime(x):
	temp = sigmoid(x)
	return temp*(1-temp)

def tanh_prime(x):
	temp =  np.tanh(x)
	return (1-temp)(1+temp)	