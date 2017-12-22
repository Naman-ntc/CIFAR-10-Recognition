import numpy as np
from layers import *
from losses import *

class FullyConnectedNet(object):

	def __init__(self,X,y,Xv,yv,sizes,batch_norm,drops):
		self.num_layers = len(sizes)
		self.biases = [np.random.randn(1,y)*np.sqrt(2/y) for y in sizes[1:]] ##Biases start from layer 2
		self.weights = [np.random.randn(x, y)*np.sqrt(4/(x*y)) for x, y in zip(sizes[:-1], sizes[1:])] 
		self.sizes = sizes
		self.batch_norm = batch_norm
		self.drops = drops
		self.X_train = X
		self.Y_train = y
		self.X_val = X_t
		self.Y_val = yt
		self.loss_type = 1 # from softmax-cross-entropy

	def update_batch_normalization(self,batch_norm):
		self.batch_norm = batch_norm
		#batch_norm is an list of booleans, batch_norm for this layer or not

	def updates_dropouts(self,drops):
		self.drops = drops
		#drops is an list of keep_probs, dropout factor for each layer

	def update_loss_type(self,loss):
		self.loss_type = loss

	def give_prediction(self,X):
		X = process(X)
		return np.max(self.out_vals(X),axis=1)

	def give_accuracy(self,X,y):
		X = process(X)
		Y_pred = np.max(self.out_vals(X),axis=1)
		return np.mean(Y_pred==y)

	def give_loss(self):
		x = self.out_vals(X_train)
		if (self.loss_type):
			return softmax_cross_entropy_loss(x,Y_train)[0]
		return svm_loss(x,Y_train)[0]
	
	