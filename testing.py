import load_data
from knn import KNNclassifier
import numpy

CIFAR = load_data.load_all()
data = CIFAR['data']
label = CIFAR['label']

CIFAR = load_data.load_test()
test_data = CIFAR['data']
test_label = CIFAR['label']

######################################################################################################
# For the KNN classifier

model = KNNclassifier()
model.add_data(data,label)
k_possible = [1,2,3,4,5,8,10,15,20,50,100,200,1000]
for i in range(len(k_possible)):
	print("Accuracy of KNN classifier for k = %d is %d"%(k_possible[i],model.give_prediction_and_accuracy(test_data,test_label,k_possible[i])['accuracy']))
