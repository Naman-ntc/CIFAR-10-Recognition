import numpy as np
import matplotlib.pyplot as plt

class SVMclassifier(GeneralClassifiers):
	####This is a Linear SVM without SMO optimization and kernal usage 
	def InitializePars():
		X_train = np.concatenate([np.ones((X_train.shape[0]),1),X_train],axis=1)
		n=n+1
		self.W = np.random.randn(k,n)

	def CostFunc(start,end):
		temp = np.dot(X_train,np.transpose(pars))
		temp = temp - temp[range(m),Y_train]
		temp = temp + 1
		temp[range(m),Y_train] = 0
		temp = max(0,temp)
		return temp.sum()

	def Gradient(start,end):
		temp = np.dot(X_train[start:end],np.transpose(pars))
		temp = temp - temp[range(start,end),Y_train[start:end]]
		temp = temp + 1
		temp[range(start,end),Y_train[start:end]] = 0
		temp = max(0,temp)
		correct_label_update = -1*temp.sum()
		temp[temp>0] = 1
		temp[range(start,end),Y_train[start:end]] = correct_label_update
		temp = np.dot(np.transpose(temp),X_train[start:end])
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
