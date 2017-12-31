import numpy as np
import matplotlib.pyplot as plt
from general import GeneralClassifiers

class SVMclassifier(GeneralClassifiers):
	####This is a Linear SVM without SMO optimization and kernal usage 
	def InitializePars(self):
		self.X_train = np.concatenate([np.ones(((self.X_train.shape[0]),1)),self.X_train],axis=1)
		self.n=self.n+1
		self.pars = np.random.randn(self.k,self.n)*.0001

	def CostFunc(self):
		temp = np.dot(self.X_train,np.transpose(self.pars))
		temp_ = temp[np.arange(self.m),self.Y_train.astype(int)]
		temp_ = np.reshape(temp_,(self.m,1))
		temp = temp - temp_
		temp = temp + 1
		temp[np.arange(self.m),self.Y_train.astype(int)] = 0
		temp = np.maximum(0,temp)
		return temp.sum()/self.m

	def Gradient(self,start,end):
		scores = np.dot(self.X_train[start:end],self.pars.T)
		scores_prime = scores - scores[np.arange(end-start),self.Y_train[start:end].astype(int)][:,None]
		scores_prime += 1
		scores_prime = np.maximum(scores_prime,0)
		scores_prime[np.arange(end-start),self.Y_train[start:end].astype(int)] = 0
		loss = scores_prime
		loss = np.max(loss,axis=1)
		loss = np.sum(loss)
		dpars = np.zeros_like(scores_prime)
		dpars[scores_prime>0] = 1
		dpars[np.arange(end-start),self.Y_train[start:end].astype(int)] = -1*np.sum(scores_prime,axis=1)
		dpars = np.dot(dpars.T,self.X_train[start:end])
		dpars+= self.reg*self.pars
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
		plt.savefig('SVM_Pars.png')
		plt.close()