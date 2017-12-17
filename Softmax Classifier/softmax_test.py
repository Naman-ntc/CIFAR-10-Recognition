import load_data
from softmax import Softmaxclassifier
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
best_softmax = None
learning_rates = [3e-5,8e-5,2.5e-6, 5e-6]
regularization_strengths = [0,1e-6,2e-7,8e-6]
for rs in regularization_strengths:
	for lr in learning_rates:
		model = Softmaxclassifier()
		model.add_data(data[:48000],label[:48000],data[48000:],label[48000:],10)
		model.InitializePars()
		model.GradientDescent(rs,300,lr,150)
		temp = model.Validate()
		print("For Learning Rate %.8f and regularization %d train accuracy %.4f and val accuracy %.4f"%(lr,rs,temp[0],temp[1]))
		if temp[1] > best_val:
			best_val = temp[1]
			best_softmax = model
			best = (rs,lr)           
		results[(lr,rs)] = temp[0], temp[1]

best_softmax.PlotPars()
losses = best_softmax.give_loss()
best_softmax.GradientDescent(best[0],2700,best[1],150)
plt.plot(losses)
plt.ylabel('loses over time')
plt.savefig('Losses_over_time_Softmax.png')
best_softmax.give_prediction_and_accuracy(test_data,test_label)['accuracy']