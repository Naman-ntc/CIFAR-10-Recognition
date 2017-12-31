import load_data
from neuralnetwork import NeuralNetoworkclassifier
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


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
learning_rates = [e-1,7e-1]
regularization_strengths = [0,1e-6,2e-7,8e-6]
hidden_layers = [200,300,500,1000]
for lr in learning_rates:
	for rs in regularization_strengths:
		for hd in hidden_layers:
			model = NeuralNetoworkclassifier()
			model.add_data(data[:48000],label[:48000],data[48000:],label[48000:],10)
			model.InitializePars([3072,hd,10])
			model.GradientDescent(rs,350,lr,150)
			temp = model.Validate()
			print("For Learning Rate %.8f,regularization %d and hidden layers %d train accuracy %.4f and val accuracy %.4f"%(lr,rs,hd,temp[0],temp[1]))
			if temp[1] > best_val:
				best_val = temp[1]
				#print(best_val)
				best = (rs,lr)
				best_nn = model           
			results[(lr,rs,hd)] = temp[0], temp[1]

best_nn.PlotPars()
losses = best_nn.give_loss()
best_nn.GradientDescent(best[0],3300,best[1],150)
plt.plot(losses)
plt.ylabel('loses over time')
plt.savefig('Losses_over_time_NN.png')
best_nn.give_prediction_and_accuracy(test_data,test_label)['accuracy']