import load_data
from knn import KNNclassifier
from softmax import Softmaxclassifier
from svm import SVMclassifier
from neuralnetwork import NeuralNetoworkclassifier
import numpy as np

CIFAR = load_data.load_all()
data = CIFAR['data']
label = CIFAR['label']

CIFAR = load_data.load_test()
test_data = CIFAR['data']
test_label = CIFAR['label']

######################################################################################################
# For the KNN classifier

# model = KNNclassifier()
# model.add_data(data,label)
# k_possible = [1,2,3,4,5,8,10,15,20,50,100,200,1000]
# for i in range(len(k_possible)):
# 	print("Accuracy of KNN classifier for k = %d is %d"%(k_possible[i],model.give_prediction_and_accuracy(test_data,test_label,k_possible[i])['accuracy']))

#######################################################################################################
# For the Softmax classifier

# model = Softmaxclassifier()
# model.add_data(data[:1000],label[:1000],10,1e-7,50,0.01,100)
# model.InitializePars()
# model.GradientDescent()
# print(model.give_prediction_and_accuracy(test_data[:10],test_label[:10])['accuracy'])

#######################################################################################################
# For the Softmax classifier

# model = SVMclassifier()
# model.add_data(data[:1000],label[:1000],10,1e-7,50,0.01,100)
# model.InitializePars()
# model.GradientDescent()
# print(model.give_prediction_and_accuracy(test_data[:10],test_label[:10])['accuracy'])

#######################################################################################################
# For the Neural Net classifier
np.set_printoptions(threshold=np.nan)
model = NeuralNetoworkclassifier()
model.add_data(data[:1000],label[:1000],10,1e-7,1000,0.01,100)
model.InitializePars([3072,300,10])
model.GradientDescent()
print(model.give_prediction_and_accuracy(test_data[:10],test_label[:10])['accuracy'])