import numpy as np
import matplotlib.pyplot as plt
from general import GeneralClassifiers

class NeuralNetoworkclassifier(GeneralClassifiers):

	def InitializePars(self,sizes):
		#sizes contain the number of neurons in respective layers
		#First element of sizes so obviously equals the input dimension
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.biases = [np.random.randn(1,y) for y in sizes[1:]] ##Biases start from layer 2
		self.weights = [np.random.randn(x, y) for x, y in zip(sizes[:-1], sizes[1:])]
		

	def CostFunc(self):
		temp = X_train
		for i in range(self.num_layers-1):
			bias = self.biases[i]
			weight = self.weights[i]
			temp = np.dot(temp,weight) + bias
			temp = self.sigmoid(temp)
		Y_train_temp = np.zeros((self.m,self.k))
		Y_train_temp[range(self.m),self.Y_train] = 1
		return 0.5*np.linalg.norm(Y_train_temp-temp)**2

	def Gradient(self,start,end):
		a = [None]*self.num_layers
		z = [None]*self.num_layers ##actually to be only num_layers - 1 but for sake of indexing
		# a[0] = X
		# for i in range(1,self.num_layers):
		# 	z[i] = np.dot(a[i-1],self.weights[i-1]) + self.biases[i-1]
		# 	a[i] = self.sigmoid(z[i])
		# Y = np.zeros(self.k)
		# Y[y] = 1
		# z_derivatives = [None]*num_layers
		# z_derivatives[-1] = (a[-1] - Y_train_temp)*a[-1]*(1-a[-1])
		# for i in range(2,num_layers-1):
		# 	z_derivatives[-1*i] = z_derivatives[-1*(i-i)]*np.dot((a[-1*i]*(1-a[-1*i])),W[-1*i])
		# for i in range(num_layers-1):
		# 	self.del_biases[i] +=  self.alpha*z_derivatives[i+1]/m
		# 	self.del_weights[i] +=  self.alpha*(np.dot(a[i].T,z[i+1]))/m - reg*((np.linalg.norm(weights[i]))**2)
		a[0] = self.X_train[start:end]
		for i in range(1,self.num_layers):
			z[i] = np.dot(a[i-1],self.weights[i-1]) + self.biases[i-1]
			a[i] = self.sigmoid(z[i])
			#print("Layer %d =>"%(i)) 
			#print(self.biases[i-1])
			#print(a[i][0,:])			
		#print(a[0][0,:])
		# print(self.weights[0][:,0])
		# print(np.dot(a[0][0,:],self.weights[0][:,0]))
		# print(self.biases[0][0,0])
		# print(z[1][0,0])
		Y_train_temp = np.zeros((self.mini_batch_size,self.k))
		Y_train_temp[range(self.mini_batch_size),self.Y_train[start:end].astype(int)] = 1
		z_derivatives = [None]*self.num_layers
		z_derivatives[-1] = (a[-1] - Y_train_temp)*a[-1]*(1-a[-1])/self.mini_batch_size
		for i in range(2,self.num_layers):
			z_derivatives[-1*i] = np.dot(z_derivatives[-1*(i-1)],self.weights[-1*(i-1)].T)*(a[-1*i]*(1-a[-1*i]))
		for i in range(self.num_layers-1):
			self.del_biases[i] +=  self.alpha * z_derivatives[i+1].sum(axis=0)
			self.del_weights[i] +=  self.alpha*(np.dot(a[i].T,z_derivatives[i+1])) + self.reg*(self.weights[i])

	def GradientDescent(self):
		start = 0
		for i in range(self.epochs):
			#print("Epoch %d"%(i))
			#print(self.weights[0][0,:]*self.X_train[0,:] + self.biases[0])
			self.del_biases = [np.zeros((1,y)) for y in self.sizes[1:]] ##Biases start from layer 2
			self.del_weights = [np.zeros((x, y)) for x, y in zip(self.sizes[:-1], self.sizes[1:])]
			# for i in range(start,min(m,start+mini_batch_size)):
			# 	Gradient(X_train[i],Y_train[i])
			self.Gradient(start,min(self.m,start+self.mini_batch_size))
			for i in range(self.num_layers-1):
				self.biases[i] = self.biases[i] - self.del_biases[i]
				self.weights[i] = self.weights[i] - self.del_weights[i]
			
			start = (start+self.mini_batch_size)%self.m
		return

	def predict(self):
		temp = self.X_test
		for i in range(self.num_layers-1):
			bias = self.biases[i]
			weight = self.weights[i]
			temp = np.dot(temp,weight) + bias
			temp = self.sigmoid(temp)
		return np.argmax(temp,axis=1)

	def sigmoid(self,z):
		return 1.0/(1.0+np.exp(-z))

	def sigmoid_prime(self,z):
		return sigmoid(z)*(1-sigmoid(z))