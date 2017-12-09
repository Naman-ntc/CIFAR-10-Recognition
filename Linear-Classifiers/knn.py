import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from scipy.stats import mode


#Implementing a general K nearest neighbours algorithm

class KNNclassifier(object):

	def __init__(self):
		pass

	def add_data(self,X,Y):
		#These X and Y are the train sets
		self.X_train = X
		self.Y_train = Y
		return

	def	give_prediction(self,X,k):
		#These X and Y are the test sets
		#Returns a prediction of what corresponding X to each element of X the label should be
		self.X_test = X
		self.k = k
		return self.predict()

	def give_prediction_and_accuracy(self,X,Y,k):
		#These X and Y are the test sets
		#Returns a prediction of what corresponding X to each element of X the label should be and also
		#analysis of the prediction with respect to true Y
		self.X_test = X
		self.Y_test = Y
		self.k = k
		Y_predict = self.predict()
		accuracy = self.find_error(Y_predict)/x.shape[0]
		return {'prediction' : Y_predict,'accuracy' : accuracy}

	def update_k(self,k):
		#Updates the k ok 'k' nearest neighbours
		self.k = k
	
	def predict(self):
		dists = cdist(self.X_test,self.X_train)
		idx = np.argpartition(dists, self.k, axis=1)[:,:self.k]
		nearest_dists = np.take(self.Y_train, idx)
		out = mode(nearest_dists,axis=1)
		return out[0]

	def predict_using_KDTree(self):
		tree = KDTree(self.X_train,leafsize=self.X_train.shape[0]+1)
		idx = tree.query(self.X_test,self.k)[1]
		nearest_dists = np.take(self.Y_train, idx)
		out = mode(nearest_dists,axis=1)
		return out[0]

	def find_error(self,prediction):
		return np.count_nonzero(prediction==self.Y_test)
