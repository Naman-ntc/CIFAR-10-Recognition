import load_data
from knn import KNNclassifier
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

CIFAR = load_data.load_all()
data = CIFAR['data']
label = CIFAR['label']
label = np.asarray(label)
label = np.reshape(label,(label.shape[0],1))

CIFAR = load_data.load_test()
test_data = CIFAR['data']
test_label = CIFAR['label']
test_label = np.asarray(test_label)
test_label = np.reshape(test_label,(test_label.shape[0],1))


model = KNNclassifier()
k_possible = [1,2,3,4,5,8,10,15,20,25,50,75,100,200,250,500,1000,5000]
acc_for_k = dict(zip(k_possible,[[]]*13))

for j in range(5):
	train = np.random.permutation(50000)
	train = train[:8000].astype(int)
	test = np.random.permutation(10000)
	test = test[:500].astype(int)
	model.add_data(data[train],label[train])
	for i in k_possible:
		Accuracy = model.give_prediction_and_accuracy(test_data[test],test_label[test],i)['accuracy']
		acc_for_k[i].append(Accuracy)


# plot the raw observations
for k in k_possible:
  accuracies = acc_for_k[k]
  plt.scatter([k] * len(accuracies), accuracies)
  print("For k = %d the accuracy mean is %f"%(k,np.mean(accuracies)) )

# plot the trend line with error bars that correspond to standard deviation
accuracies_mean = np.array([np.mean(v) for k,v in sorted(acc_for_k.items())])
accuracies_std = np.array([np.std(v) for k,v in sorted(acc_for_k.items())])
plt.errorbar(k_possible, accuracies_mean, yerr=accuracies_std)
plt.title('Cross-validation on k')
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.show()
plt.savefig('KNN-plot.png')