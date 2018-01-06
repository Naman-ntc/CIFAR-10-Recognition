# CIFAR-10-Recognition
Image Recognition on the CIFAR-10 training set using various approaches like Convolutional networks, Support Vector Machines, Softmax regression, Nearest Neighbours using only Numpy. Also used Tensorflow to build convNets

Download the CIFAR-10 dataset by executing the script get_data.sh.
Alternatively you can download the data from [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)

### CIFAR-10 Dataset
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 
The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class. 

### Algorithms and results

|			Algorithm 								|    Results  (Accuracy)		|
|---------------------------------------------------|-------------------------------|
| [Nearest Neighbours](K_Nearest_Neighbours) 		|	 28.15%						|
| [Support Vector Machines](Linear_SVM)				|	 36.7%						|
| [Softmax Regression](Softmax_Classifier)			|	 37.4%						|
| [FeedForward Neural Networks](FeedForward_Neural_Network_classifier)|  44.2% (Only 3 layers) 	 |
| [Convolutional Neural Networks](Modular-CNN's)    |    							|
| [Neural Networks on Tensorflow](Tensorflow)		|	68.3%						|	

### References
Most of the work was inspired from CS231n assignments and was completely authored by me
