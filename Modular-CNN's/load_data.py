import pickle
import numpy

main = "../cifar-10-python/cifar-10-batches-py/data_batch_"

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def load_all():
	data = numpy.zeros((50000,3072))
	label = numpy.zeros(50000)
	for i in range(5):
		temp = unpickle(main+str(i+1))
		data[10000*i:10000*(i+1),:] = temp[b'data']
		label[10000*i:10000*(i+1)] = temp[b'labels']
	return {'data':data,'label':label}

def load_test():
	temp = unpickle(main[:-11]+"test_batch")
	data= temp[b'data']
	label = temp[b'labels']
	return {'data':data,'label':label}


#####################################################
# Uncomment to load images

#CIFAR = load_all()
#data = CIFAR['data']
#label = CIFAR['label']
#####################################################

#####################################################
# Uncomment either first or second to store rgb matrix

#rgb = data[1].reshape(3,32,32).transpose()

#r = data[0,0:1024].reshape(32,32)
#g = data[0,1024:2048].reshape(32,32)
#b = data[0,2048:3072].reshape(32,32)
#rgb = np.dstack((r,g,b))
#####################################################

#####################################################
# Uncomment to watch the images formed

#plt.imshow(rgb)
#plt.show()
#####################################################


#####################################################
# This is a bit of magic to make matplotlib figures appear inline in the notebook
# rather than in a new window.

#plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
#plt.rcParams['image.interpolation'] = 'nearest'
#plt.rcParams['image.cmap'] = 'gray'

#####################################################


#####################################################
# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
# Uncomment to visualize in awesome manner!
# classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
# num_classes = len(classes)
# samples_per_class = 7
# for y, cls in enumerate(classes):
#     idxs = np.flatnonzero(Y_train == y)
#     idxs = np.random.choice(idxs, samples_per_class, replace=False)
#     for i, idx in enumerate(idxs):
#         plt_idx = i * num_classes + y + 1
#         plt.subplot(samples_per_class, num_classes, plt_idx)
#         plt.imshow(X_train[idx].astype('uint8').reshape(3,32,32).swapaxes(0,-1))
#         plt.axis('off')
#         if i == 0:
#             plt.title(cls)
# plt.show()
####################################################
