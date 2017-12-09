import load_data
from neuralnetwork import NeuralNetoworkclassifier
import matplotlib.pyplot as plt
import numpy as np

CIFAR = load_data.load_all()
data = CIFAR['data']
label = CIFAR['label']
label = np.asarray(label)

CIFAR = load_data.load_test()
test_data = CIFAR['data']
test_label = CIFAR['label']
test_label = np.asarray(test_label)



results = {}
best_val = -1
best_nn = None
learning_rates = [3e-7,8e-7,2.5e-8, 3e-8]
regularization_strengths = [0,1e-6,2e-7,8e-6]
hidden_layers = [200,300,500,1000]
for lr in learning_rates:
	for rs in regularization_strengths:
		for hd in hidden_layers:
			model = NeuralNetoworkclassifier()
			model.add_data(data[:48000],label[:48000],data[48000:],label[48000:],10)
			model.InitializePars([3072,hd,10])
			model.GradientDescent(rs,3200,lr,150)
			temp = model.Validate()
			print("For Learning Rate %f,regularization %d and hidden layers %d train accuracy %f and val accuracy %f"%(lr,rs,hd,temp[0],temp[1]))
			if temp[1] > best_val:
				best_val = temp[1]
				print(best_val)
				best_svm = model           
				results[(lr,rs,hd)] = temp[0], temp[1]

best_svm.PlotPars()
losses = best_svm.give_loss()
plt.plot(losses)
plt.ylabel('loses over time')
plt.savefig('Losses_over_time_SVM.png')
best_svm.give_prediction_and_accuracy(test_data,test_label)['accuracy']