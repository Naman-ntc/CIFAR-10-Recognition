import numpy as np
import matplotlib.pyplot as plt
from general import GeneralClassifiers

class Softmaxclassifier(GeneralClassifiers):
	
	def InitializePars(self):
		self.X_train = np.concatenate([np.ones(((self.X_train.shape[0]),1)),self.X_train],axis=1)
		self.n=self.n+1
		self.pars = np.random.randn(self.k,self.n)*0.0001

	def CostFunc(self):
		x = np.dot(self.X_train,np.transpose(self.pars))
		x = x - np.max(x)
		
		N,K = x.shape
		
		log_probabilities = x - np.log(np.sum(np.exp(x),axis=1,keepdims=True))
		loss = -np.sum(log_probabilities[np.arange(N),self.Y_train[start:end].astype(int)])
		
		return loss/self.m

	def Gradient(self,start,end):
		x = self.X_train[start:end]
		x = np.dot(x,np.transpose(self.pars))
		x = x - np.max(x,axis=1,keepdims=True)
		N,K = x.shape
		
		log_probabilities = x - np.log(np.sum(np.exp(x),axis=1,keepdims=True))
		loss = -np.sum(log_probabilities[np.arange(N),self.Y_train[start:end].astype(int)])
		loss/=N

		probabilities = np.exp(log_probabilities)
		dx = probabilities
		dx[np.arange(N),self.Y_train[start:end].astype(int)] -=1
		dpars = np.dot(self.X_train[start:end].T,dx).T
		dpars += self.reg*self.pars
		return (dpars,loss)
	
	def set_lr(self,alpha):
		self.alpha = alpha

	def set_reg(self,reg):	
		self.reg = reg

	def set_bs(self,mini_batch_size):
		self.mini_batch_size = mini_batch_size

	def GradientDescent(self,epochs):
		self.epochs = epochs
		self.loss_with_epoch = [None]*epochs
		start = 0
		for i in range(self.epochs):
			temp = self.Gradient(start,min(self.m,start+self.mini_batch_size))
			self.loss_with_epoch[i] = temp[1]
			#print(temp[1])
			self.pars += -1*self.alpha*temp[0]
			start = (start+self.mini_batch_size)%self.m
			self.alpha*=0.95
		return

	def predict(self):
		self.X_test = np.concatenate([np.ones(((self.X_test.shape[0]),1)),self.X_test],axis=1)
		temp = np.dot(self.X_test,np.transpose(self.pars))
		return np.argmax(temp,axis=1)	
	
	def PlotPars(self):
		w = np.reshape(self.pars[:,1:],(32,32,3,10))
		w_min, w_max = np.min(w), np.max(w)

		classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
		for i in range(10):
			plt.subplot(2, 5, i + 1)
  
  			# Rescale the weights to be between 0 and 255
			wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
			plt.imshow(wimg.astype('uint8'))
			plt.axis('off')
			plt.title(classes[i])
		
		plt.savefig('Softmax_Pars.png')
		plt.close()	