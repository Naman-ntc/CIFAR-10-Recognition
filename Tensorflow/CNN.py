import load_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.reset_default_graph()

#######################################################################################################	

CIFAR = load_data.load_all()
data = CIFAR['data']
label = CIFAR['label']
label = np.asarray(label)

CIFAR = load_data.load_test()
test_data = CIFAR['data']
test_label = CIFAR['label']
test_label = np.asarray(test_label)


#######################################################################################################	

self.data = np.reshape(self.data,(self.num_training+self.num_validation,32,32,3))
mask = range(self.num_training, self.num_training + self.num_validation)
self.X_val = self.data[mask]
self.Y_val = self.label[mask]
mask = range(self.num_training)
self.X_train = self.data[mask]
self.Y_train = self.label[mask]

		
self.mean_image = np.mean(self.X_train, axis=0)
self.X_train -= self.mean_image
self.X_val -= self.mean_image

self.test_data = np.reshape(self.test_data,(self.num_test,32,32,3))
mask = range(self.num_test)
self.X_test = self.test_data[mask]
self.Y_test = self.test_label[mask]

self.X_test -= self.mean_image
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
temp8 = (tf.matmul(temp7, W3)
B3 = tf.Variable(tf.random_normal([10]))
temp8 = tf.bias_add(temp8,B3)

######################################################################################################

y_out = temp8

total_loss = tf.losses.hinge_loss(tf.one_hot(y,10),logits=y_out)
mean_loss = tf.reduce_mean(total_loss)

# define our optimizer
optimizer = tf.train.AdamOptimizer(5e-4) # select optimizer and set learning rate
train_step = optimizer.minimize(mean_loss)

#####################################################################################################

batch_size = 100
epoches = 2
sess = tf.Session()

####################################################################################################

for i in range(epoches):
	rand_index = np.random.choice(num_training, size=batch_size)
    print(sess.run(mean_loss,feed_dict={X:X_train[rand_index],y=Y_train[rand_index]}))