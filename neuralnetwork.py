import numpy as np
import matplotlib.pyplot as plt

class NeuralNetoworkclassifier(GeneralClassifiers):

	def InitializePars(self,sizes):
		#sizes contain the number of neurons in respective layers
		#First element of sizes so obviously equals the input dimension
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.biases = [np.random.randn(y, 1) for y in sizes[1:]] ##Biases start from layer 2
		self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

	def 	

	def predict():
		temp = np.transpose(X_train)
		for i in range(num_layers-1):
			bias = biases[i]
			weight = weights[i]
			temp = np.dot(weight,temp) + bias
			temp = sigmoid(temp)
		temp = np.transpose(temp)
		return np.argmax(temp,axis=1)
	