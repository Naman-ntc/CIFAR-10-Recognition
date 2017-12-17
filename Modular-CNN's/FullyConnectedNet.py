import numpy as np

class FullyConnectedNet(object):

	def __init__(self,sizes,batch_norm,drops):
		self.num_layers = len(sizes)
		self.biases = [np.random.randn(1,y)*np.sqrt(2/y) for y in sizes[1:]] ##Biases start from layer 2
		self.weights = [np.random.randn(x, y)*np.sqrt(4/(x*y)) for x, y in zip(sizes[:-1], sizes[1:])] 
		self.sizes = sizes
		self.batch_norm = batch_norm
		self.drops = drops

	def update_batch_normalization(self,batch_norm):
		self.batch_norm = batch_norm
		#batch_norm is an array of booleans, batch_norm for this layer or not

	def updates_dropouts(self,drops):
		self.drops = drops			