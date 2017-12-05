import numpy as np
import matplotlib.pyplot as plt

class Softmaxclassifier(GeneralClassifiers):
	
	def InitializePars():
		X_train = np.concatenate([np.ones((X_train.shape[0]),1),X_train],axis=1)
		n=n+1
		self.pars = np.random.randn(k,n)

	def CostFunc():
		temp = np.dot(X_train,np.transpose(pars))
		temp = temp - np.max(temp)
		temp = np.exp(temp)
		temp = temp/np.sum(temp,axis=1, keepdims=True) ##or np.sum(temp,axis=1)[0:None]
		temp = X_train[np.arange(m),Y_train.astype(int)]
		temp = temp.sum()
		return temp

	def Gradient(start,end):
		temp = np.dot(X_train[start:end],np.transpose(pars))
		temp = np.exp(temp)
		temp = temp/np.sum(temp,axis=1, keepdims=True) ##or np.sum(temp,axis=1)[0:None]
		temp_ = np.zeros((end-start,k))
		temp_[np.arange(m),Y_train[start:end].astype(int)] = 1
		temp = temp - temp_
		temp = np.dot(temp.T,X_train[start:end])
		return temp

	def GradientDescent():
		start = 0
		for i in range(epochs):
			pars += -1*alpha*Gradient(start,min(m,start+mini_batch_size))
			start = (start+mini_batch_size)%m
		return

	def predict():
		X_test = np.concatenate([np.ones((X_test.shape[0]),1),X_test],axis=1)
		temp = np.dot(X_test,np.transpose(pars))
		return np.argmax(temp,axis=1)	
	
	#def PlotPars():
