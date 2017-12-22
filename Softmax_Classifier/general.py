import numpy as np
import matplotlib.pyplot as plt


class GeneralClassifiers(object):
	def __init__(self):
		pass

	def add_data(self,X=None,Y=None,valX=None,valY=None,num_classes=None): 
		#These X and Y are the training sets
		#Dimensions of X are as m x n
		self.X_train = X
		self.Y_train = Y
		self.X_train_orig = X
		self.Y_train_orig = Y
		self.X_val = valX
		self.Y_val = valY
		self.m = self.X_train.shape[0]
		self.m_val = self.X_val.shape[0]
		self.n = self.X_train.shape[1]
		self.k = num_classes
		self.StandardizeTrain()
		return

	def InitializePars():
		pass	
	
	def StandardizeTrain(self):
		self.train_mean = np.reshape(np.mean(self.X_train,axis=0),(1,self.n))
		self.X_train = self.X_train - self.train_mean
		#self.train_std = np.std(self.X_train,axis=0)
		#self.X_train = self.X_train/self.train_std
		self.X_val = self.X_val - self.train_mean
		#self.X_val = self.X_val/self.train_std
		return

	def StandardizeTest(self):
		self.X_test = self.X_test - self.train_mean
		#self.X_test = self.X_test/self.train_std
		return

	def CostFunc(self):
		pass

	def Gradient(self,start,end):
		pass

	def GradientDescent(start,end):
		pass
			
	def	give_prediction(self,X): 
		#These X and Y are the test sets
		#Returns a prediction of what corresponding X to each element of X the label should be
		self.X_test = X
		#self.StandardizeTest()
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

	def Validate(self):
		train_accuracy = self.give_prediction_and_accuracy(self.X_train_orig,self.Y_train_orig)['accuracy']	
		validate_accuracy = self.give_prediction_and_accuracy(self.X_val,self.Y_val)['accuracy']
		return (train_accuracy,validate_accuracy)
			
	def find_error(self,prediction):
		return np.mean(prediction==self.Y_test)	

	def give_loss(self):
		return self.loss_with_epoch	