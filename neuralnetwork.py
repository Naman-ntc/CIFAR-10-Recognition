import numpy as np
import matplotlib.pyplot as plt

class NeuralNetoworkclassifier(GeneralClassifiers):

	def InitializePars(self,sizes):
		#sizes contain the number of neurons in respective layers
		#First element of sizes so obviously equals the input dimension
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.biases = [np.random.randn(y, 1) for y in sizes[1:]] ##Biases start from layer 2
		self.weights = [np.random.randn(x, y) for x, y in zip(sizes[:-1], sizes[1:])]

	def CostFunc():
		temp = X_test
		for i in range(num_layers-1):
			bias = biases[i]
			weight = weights[i]
			temp = np.dot(temp,weight) + bias
			temp = sigmoid(temp)
		Y_train_temp = np.zeros((m,k))
		Y_train_temp[range(m),Y_train] = 1
		return 0.5*np.linalg.norm(Y_train_temp-temp)**2

	def Gradient(X,y):
		a = [None]*num_layers
		z = [None]*num_layers ##actually to be only num_layers - 1 but for sake of indexing
		a[0] = X
		for i in range(1,num_layers):
			z[i] = np.dot(a[i-1],W[i-1]) + b[i-1]
			a[i] = sigmoid(z[i])
		Y = np.zeros(k)
		Y[y] = 1
		z_derivatives = [None]*num_layers
		z_derivatives[-1] = (a[-1] - Y_train_temp)*a[-1]*(1-a[-1])
		for i in range(2,num_layers-1):
			z_derivatives[-1*i] = z_derivatives[-1*(i-i)]*np.dot((a[-1*i]*(1-a[-1*i])),W[-1*i])
		for i in range(num_layers-1):
			del_biases[i] +=  alpha*z_derivatives[i+1]
			del_weights[i] +=  alpha*(np.dot(a[i].T,z[i+1])) - reg*((np.linalg.norm(weights[i]))**2)

	def GradientDescent():
		start = 0
		for i in range(epochs):
			self.del_biases = [np.zeros((y, 1)) for y in sizes[1:]] ##Biases start from layer 2
			self.del_weights = [np.zeros((x, y)) for x, y in zip(sizes[:-1], sizes[1:])]
			for i in range(start,min(m,start+mini_batch_size)):
				Gradient(X_train[start],Y_train[start])
			for i in range(num_layers-1):
				biases[i] = biases[i] - del_biases[i]
				weights[i] = weights[i] - del_weights[i]		
			start = (start+mini_batch_size)%m
		return

	def predict():
		temp = X_test
		for i in range(num_layers-1):
			bias = biases[i]
			weight = weights[i]
			temp = np.dot(temp,weight) + bias
			temp = sigmoid(temp)
		return np.argmax(temp,axis=1)

	def sigmoid(z):
		return 1.0/(1.0+np.exp(-z))

	def sigmoid_prime(z):
		return sigmoid(z)*(1-sigmoid(z))
