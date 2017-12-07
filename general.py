import numpy as np
import matplotlib.pyplot as plt


class GeneralClassifiers(object):
	def __init__(self):
		pass

	def add_data(self,X,Y,num_classes,reg,epochs,alpha,mini_batch_size): 
		#These X and Y are the training sets
		#Dimensions of X are as m x n
		self.X_train = X
		self.Y_train = Y
		self.m = self.X_train.shape[0]
		self.n = self.X_train.shape[1]
		self.k = num_classes
		self.reg = reg
		self.epochs = epochs
		self.alpha = alpha
		self.mini_batch_size = mini_batch_size
		self.StandardizeTrain()
		return

	def InitializePars():
		pass	

	def UpdateReg(self,reg):
		self.reg = reg	
		return 

	def UpdateEpochs(self,epochs):
		self.epochs = epochs
		return

	def UpdateAlpha(self,alpha):
		self.alpha = alpha	
		return

	def UpdataMiniBatchSize(self,mini_batch_size):
		self.mini_batch_size = mini_batch_size

	def StandardizeTrain(self):
		mean = np.mean(self.X_train,axis=0)
		self.X_train = self.X_train - mean
		var = np.std(self.X_train,axis=0)
		self.X_train = self.X_train/var
		return

	def StandardizeTest(self):
		mean = np.mean(self.X_test,axis=0)
		self.X_test = self.X_test - mean
		var = np.std(self.X_test,axis=0)
		self.X_test = self.X_test/var
		return

	def CostFunc():
		pass

	def Gradient():
		pass

	def GradientDescent(start,end):
		pass
			
	def	give_prediction(self,X): 
		#These X and Y are the test sets
		#Returns a prediction of what corresponding X to each element of X the label should be
		self.X_test = X
		self.StandardizeTest()
		return self.predict()

	def predict():
		pass	

	def give_prediction_and_accuracy(self,X,Y):	
		#These X and Y are the test sets
		#Returns a prediction of what corresponding X to each element of X the label should be and also
		#analysis of the prediction with respect to true Y
		self.X_test = X
		self.Y_test = Y
		self.StandardizeTest()
		Y_predict = self.predict()
		accuracy = self.find_error(Y_predict)
		return {'prediction' : Y_predict,'accuracy' : accuracy}

	def find_error(self,prediction):
		return np.count_nonzero(prediction==self.Y_test)			