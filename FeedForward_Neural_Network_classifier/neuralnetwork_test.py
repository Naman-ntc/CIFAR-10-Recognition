import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import load_data as loader
from neuralnetwork import NeuralNetoworkclassifier
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

data, label = loader.load_all()
test_data, test_label = loader.load_test()

results = {}
best_val = -1
best_nn = None
learning_rates = [1e-3,1e-4,1e-5,1e-6,1e-2]
regularization_strengths = [0]
hidden_layers = [100,200]

for rs in regularization_strengths:	
	for lr in learning_rates:	
		for hd in hidden_layers:
			model = NeuralNetoworkclassifier()
			model.add_data(data[:480],label[:480],data[48000:],label[48000:],10)
			model.InitializePars([3072,hd,10])
			model.set_lr(lr)
			model.set_reg(rs)
			model.set_bs(600)
			model.GradientDescent(350)
			temp = model.Validate()
			print("For Learning Rate %.8f,regularization %d and hidden layers %d train accuracy %.4f and val accuracy %.4f"%(lr,rs,hd,temp[0],temp[1]))
			if temp[1] > best_val:
				best_val = temp[1]
				best_nn = model
				best = (lr,rs,hd)           
			results[(lr,rs,hd)] = temp

print("\n\nBest Validation Accuracy %.4f for lr :%.8f rs :%.4f hd :%f "%(best_val,best[0],best[1],best[2]))
losses = best_nn.give_loss()
best_nn.GradientDescent(1200)
plt.plot(losses)
plt.ylabel('loses over time')
plt.savefig('Losses_over_time_NN.png')
print("And Testing Accuracy is %.4f"%(best_nn.give_prediction_and_accuracy(test_data,test_label)['accuracy']))