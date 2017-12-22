import load_data as loader
from softmax import Softmaxclassifier
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

data, label = loader.load_all()
test_data, test_label = loader.load_test()

results = {}
best_val = -1
best_softmax = None
learning_rates = [3e-6,2e-7,8e-7,6e-8,9e-6,4e-5]
regularization_strengths = [0,1,10,1e2,1e3,1e4,1e5,1e6,1e-1,1e-2,1e-3,1e-4]
batch_sizes = [32,64,128,150,300]
for rs in regularization_strengths:
	for lr in learning_rates:
		for bs in batch_sizes:
			model = Softmaxclassifier()
			model.add_data(data[:48000],label[:48000],data[48000:],label[48000:],10)
			model.InitializePars()
			model.set_lr(lr)
			model.set_reg(rs)
			model.set_bs(bs)
			model.GradientDescent(150)
			temp = model.Validate()
			print("For lr %.8f ,rs %.4f and bs %f Train Accuracy %.4f and Validation Accuracy %.4f"%(lr,rs,bs,temp[0],temp[1]))
			
			if temp[1] > best_val:
				best_val = temp[1]
				best_softmax = model
			results[(lr,rs,bs)] = temp

print("\n\nBest Validation Accuracy %.4f for lr :%.8f rs :%.4f bs :%f "%(best_val,best[0],best[1],best[2]))
best_softmax.PlotPars()
losses = best_softmax.give_loss()
best_softmax.GradientDescent(750)
losses = losses + best_softmax.give_loss()
plt.plot(losses)
plt.ylabel('loses over time')
plt.savefig('Losses_over_time_Softmax.png')
plt.close()
print("And Testing Accuracy is %.4f"%(best_softmax.give_prediction_and_accuracy(test_data,test_label)['accuracy']))