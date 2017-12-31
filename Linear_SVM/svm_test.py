import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import load_data as loader
from svm import SVMclassifier
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

data, label = loader.load_all()
test_data, test_label = loader.load_test()

results = {}
best_val = -1
best_svm = None
learning_rates = [1e-6,5e-6,1e-5,7e-5,3e-4,6e-4,1e-3]
regularization_strengths = [0,1,10,1e2,1e3,1e4,1e5,1e6,1e7,1e-1,1e-2,1e-3]
batch_sizes = [32,64,128,150,300,600,1000]
for lr in learning_rates:
	for rs in regularization_strengths:
		for bs in batch_sizes:
			model = SVMclassifier()
			model.add_data(data[:48000],label[:48000],data[48000:],label[48000:],10)
			model.InitializePars()
			model.set_lr(lr)
			model.set_reg(rs)
			model.set_bs(bs)
			model.GradientDescent(750)
			temp = model.Validate()
			print("For lr %.8f ,rs %.4f and bs %f Train Accuracy %.4f and Validation Accuracy %.4f"%(lr,rs,bs,temp[0],temp[1]))			

			if temp[1] > best_val:
				best_val = temp[1]
				best_svm = model        
				best = (lr,rs,bs)     
			results[(lr,rs,bs)] = temp

print("\n\nBest Validation Accuracy %.4f for lr :%.8f rs :%.4f bs :%f "%(best_val,best[0],best[1],best[2]))
best_svm.PlotPars()
losses = best_svm.give_loss()
best_svm.GradientDescent(750)
losses = losses + best_svm.give_loss()
plt.plot(losses[10:])
plt.ylabel('loses over time')
plt.savefig('Losses_over_time_SVM.png')
plt.close()
print("And Testing Accuracy is %.4f"%(best_svm.give_prediction_and_accuracy(test_data,test_label)['accuracy']))
