import numpy as np
import matplotlib.pyplot as plt
from general import GeneralClassifiers

class SVMclassifier(GeneralClassifiers):
	####This is a Linear SVM without SMO optimization and kernal usage 
	def InitializePars(self):
		self.X_train = np.concatenate([np.ones(((self.X_train.shape[0]),1)),self.X_train],axis=1)
		self.n=self.n+1
		self.pars = np.random.randn(self.k,self.n)

	def CostFunc(start,end):
		temp = np.dot(self.X_train,np.transpose(pars))
		temp = temp - temp[range(self.m),Y_train]
		temp = temp + 1
		temp[range(self.m),self.Y_train] = 0
		temp = max(0,temp)
		return temp.sum()

	def Gradient(self,start,end):
		temp = np.dot(self.X_train[start:end],np.transpose(self.pars))
		temp_ = temp[np.arange(self.mini_batch_size),self.Y_train[start:end].astype(int)]
		for i in range(self.k):
			temp[:,i] = temp[:,i] - temp_
		temp = temp + 1
		temp[np.arange(self.mini_batch_size),self.Y_train[start:end].astype(int)] = 0
		temp = np.maximum(0,temp)
		correct_label_update = -1*temp.sum()
		temp[temp>0] = 1
		temp[np.arange(self.mini_batch_size),self.Y_train[start:end].astype(int)] = correct_label_update
		temp = np.dot(np.transpose(temp),self.X_train[start:end])
		return temp

	def GradientDescent(self):
		start = 0
		for i in range(self.epochs):
			self.pars += -1*self.alpha*self.Gradient(start,min(self.m,start+self.mini_batch_size))/self.mini_batch_size
			self.pars += self.reg*self.pars
			start = (start+self.mini_batch_size)%self.m
		return

	def predict(self):
		self.X_test = np.concatenate([np.ones(((self.X_test.shape[0]),1)),self.X_test],axis=1)
		temp = np.dot(self.X_test,np.transpose(self.pars))
		return np.argmax(temp,axis=1)
