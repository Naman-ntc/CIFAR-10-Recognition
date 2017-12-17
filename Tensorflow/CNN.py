import load_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#xtf.reset_default_graph()

#######################################################################################################	

CIFAR = load_data.load_all()
data = CIFAR['data']
label = CIFAR['label']
label = np.asarray(label)

CIFAR = load_data.load_test()
test_data = CIFAR['data']
test_label = CIFAR['label']
test_label = np.asarray(test_label)

#####################################################################################################

batch_size = 100
epoches = 2

num_training=40000
num_validation=10000
num_test=10000

#######################################################################################################	

data = np.reshape(data,(num_training+num_validation,32,32,3))
mask = range(num_training, num_training + num_validation)
X_val = data[mask]
Y_val = label[mask]
mask = range(num_training)
X_train = data[mask]
Y_train = label[mask]

		
mean_image = np.mean(X_train, axis=0)
X_train -= mean_image
X_val -= mean_image

test_data = np.reshape(test_data,(num_test,32,32,3))
mask = range(num_test)
X_test = test_data[mask]
Y_test = test_label[mask]

X_test -= mean_image
#######################################################################################################

X = tf.placeholder(tf.float32,shape=[None,32,32,3])
y = tf.placeholder(tf.int,shape=[None])
N = X.shape[0]
W1 = tf.Variable(tf.random_normal([7,7,3,32]))
W2 = tf.Variable(tf.random_normal([3,3,32,25]))
temp1 = tf.nn.conv2d(X,W1,strides=[1, 1, 1, 1],padding='VALID')
D = temp1.shape
B1 = tf.Variable(tf.random_normal(D[1:]))
temp1 = tf.nn.bias_add(temp1,B1)
temp2 = tf.nn.leaky_relu(temp1,alpha=0.05)
temp3 = tf.nn.dropout(temp2)
temp4 = tf.nn.conv2d(temp3,W2,strides=[1, 2, 2, 1],padding='VALID')
D = temp4.shape
B2 = tf.Variable(tf.random_normal(D[1:]))
temp4 = tf.nn.bias_add(temp4,B2)
temp5 = tf.nn.leaky_relu(temp1,alpha=0.1)
temp6 = tf.nn.max_pool(temp5,[1,2,2,1],strides=[1,1,1,1],padding='VALID')
D = temp6.shape
D = D[1]*D[2]*D[3]
temp7 = tf.reshape(temp6,[N,D])
W3 = tf.Variable(tf.random_normal([D,10]))
temp8 = tf.matmul(temp7, W3)
B3 = tf.Variable(tf.random_normal([10]))
temp8 = tf.bias_add(temp8,B3)

######################################################################################################

y_out = temp8

total_loss = tf.losses.hinge_loss(tf.one_hot(y,10),logits=y_out)
mean_loss = tf.reduce_mean(total_loss)

# define our optimizer
optimizer = tf.train.AdamOptimizer(5e-4) # select optimizer and set learning rate
train_step = optimizer.minimize(mean_loss)

####################################################################################################

with tf.Session() as sess :
	with tf.device('/cpu:0'):
		for i in range(epoches):
			rand_index = np.random.choice(num_training, size=batch_size)
			print(sess.run(mean_loss,feed_dict={X:X_train[rand_index],y:Y_train[rand_index]}))