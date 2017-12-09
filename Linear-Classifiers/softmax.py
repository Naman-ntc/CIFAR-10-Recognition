import numpy as np
import matplotlib.pyplot as plt
from general import GeneralClassifiers

class Softmaxclassifier(GeneralClassifiers):
	
	def InitializePars(self):
		self.X_train = np.concatenate([np.ones(((self.X_train.shape[0]),1)),self.X_train],axis=1)
		self.n=self.n+1
		self.pars = np.random.randn(self.k,self.n)*0.001

	def CostFunc(self):
		temp = np.dot(self.X_train,np.transpose(self.pars))
		temp = temp - np.max(temp)
		temp = np.exp(temp)
		temp = temp/np.sum(temp,axis=1, keepdims=True) ##or np.sum(temp,axis=1)[0:None]
		temp = temp[np.arange(self.m),list(self.Y_train.astype(int))]
		
		temp = np.log(np.prod(temp))
		return -1*temp/self.m

	def Gradient(self,start,end):
		temp = np.dot(self.X_train[start:end],np.transpose(self.pars))
		temp = temp - np.reshape(np.max(temp,axis=1),(temp.shape[0],1))
		temp = np.exp(temp)
		temp = temp/np.sum(temp,axis=1, keepdims=True) ##or np.sum(temp,axis=1)[0:None]
		temp_ = np.zeros((end-start,self.k))
		temp_[np.arange(end-start),self.Y_train[start:end].astype(int)] = 1
		temp = temp - temp_
		temp = np.dot(temp.T,self.X_train[start:end])
		return temp/self.mini_batch_size

	def GradientDescent(self,reg,epochs,alpha,mini_batch_size):
		self.reg = reg
		self.epochs = epochs
		self.alpha = alpha
		self.mini_batch_size = mini_batch_size
		self.loss_with_epoch = [None]*epochs
		start = 0
		for i in range(self.epochs):
			self.loss_with_epoch[i] = self.CostFunc()
			self.pars += -1*self.alpha*self.Gradient(start,min(self.m,start+self.mini_batch_size))
			self.pars += -1*self.reg*self.pars
			start = (start+self.mini_batch_size)%self.m
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