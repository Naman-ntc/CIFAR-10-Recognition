import numpy as np
import matplotlib.pyplot as plt

class Softmaxclassifier(GeneralClassifiers):
	
	def InitializePars():
		self.X_train = np.concatenate([np.ones((self.X_train.shape[0]),1),self.X_train],axis=1)
		self.n=self.n+1
		self.pars = np.random.randn(k,n)

	def CostFunc():
		temp = np.dot(self.X_train,np.transpose(self.pars))
		temp = temp - np.max(temp)
		temp = np.exp(temp)
		temp = temp/np.sum(temp,axis=1, keepdims=True) ##or np.sum(temp,axis=1)[0:None]
		temp = self.X_train[np.arange(self.m),self.Y_train.astype(int)]
		temp = temp.sum()
		return temp

	def Gradient(start,end):
		temp = np.dot(self.X_train[start:end],np.transpose(pars))
		temp = np.exp(temp)
		temp = temp/np.sum(temp,axis=1, keepdims=True) ##or np.sum(temp,axis=1)[0:None]
		temp_ = np.zeros((end-start,k))
		temp_[np.arange(m),self.Y_train[start:end].astype(int)] = 1
		temp = temp - temp_
		temp = np.dot(temp.T,self.X_train[start:end])
		return temp

	def GradientDescent():
		start = 0
		for i in range(epochs):
			self.pars += -1*self.alpha*self.Gradient(start,min(self.m,start+self.mini_batch_size))
			start = (start+self.mini_batch_size)%self.m
		return

	def predict():
		self.X_test = np.concatenate([np.ones((self.X_test.shape[0]),1),self.X_test],axis=1)
		temp = np.dot(self.X_test,np.transpose(self.pars))
		return np.argmax(temp,axis=1)	
	
	#def PlotPars():
