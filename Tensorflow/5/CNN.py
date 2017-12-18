import load_data
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#xtf.reset_default_graph()
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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

batch_size = 900
epoches = 150

num_training=40000
num_validation=10000
num_test=10000


logs_path = '/home/naman/Repositories/CIFAR-10-Recognition/Tensorflow/examples/5'

#######################################################################################################	

data = np.reshape(data,(num_training+num_validation,32,32,3))
test_data = np.reshape(test_data,(num_test,32,32,3))
mask = range(num_training, num_training + num_validation)
X_val = data[mask]
Y_val = label[mask]
mask = range(num_training)
X_train = data[mask]
Y_train = label[mask]
mask = range(num_test)
X_test = test_data[mask]
Y_test = test_label[mask]

mean_image = np.mean(X_train, axis=0)
X_train -= mean_image
X_val -= mean_image
X_test -= mean_image
#######################################################################################################

X = tf.placeholder(tf.float32,shape=[None,32,32,3])
y = tf.placeholder(tf.int32,shape=[None])
train = tf.placeholder(tf.bool)
N = tf.shape(X)[0]
def test_output():
	W1 = tf.Variable(tf.random_normal([7,7,3,84]))
	W2 = tf.Variable(tf.random_normal([3,3,84,42]))
	temp1 = tf.nn.conv2d(X,W1,strides=[1, 1, 1, 1],padding='VALID')
	D = temp1.shape
	B1 = tf.Variable(tf.random_normal(D[1:]))
	temp1 = tf.add(temp1,B1)
	temp2 = tf.nn.leaky_relu(temp1,alpha=0.3)
	temp2 = tf.contrib.layers.batch_norm(temp2)
	temp3 = tf.nn.dropout(temp2,keep_prob=0.4)
	temp4 = tf.nn.conv2d(temp3,W2,strides=[1, 2, 2, 1],padding='VALID')
	D = temp4.shape
	B2 = tf.Variable(tf.random_normal(D[1:]))
	temp4 = tf.add(temp4,B2)
	#temp4 = tf.nn.dropout(temp4,keep_prob=0.6)
	temp5 = tf.nn.leaky_relu(temp4,alpha=0.2)
	temp5 = tf.contrib.layers.batch_norm(temp5)
	temp6 = tf.nn.max_pool(temp5,[1,2,2,1],strides=[1,1,1,1],padding='VALID')
	D = temp6.shape
	D = D[1]*D[2]*D[3]
	temp7 = tf.reshape(temp6,([N,D]))
	W3 = tf.Variable(tf.random_normal(tf.TensorShape([D,200])))
	temp8 = tf.matmul(temp7, W3)
	B3 = tf.Variable(tf.random_normal([200]))
	temp8 = tf.add(temp8,B3)
	temp4 = tf.nn.dropout(temp4,keep_prob=0.7)
	temp4 = tf.contrib.layers.batch_norm(temp4)
	W4 = tf.Variable(tf.random_normal(tf.TensorShape([200,10])))
	temp9 = tf.matmul(temp8, W4)
	B4 = tf.Variable(tf.random_normal([10]))
	temp9 = tf.add(temp9,B4)
	return temp9
######################################################################################################

y_out = test_output()

total_loss = tf.losses.hinge_loss(tf.one_hot(y,10),logits=y_out)
mean_loss = tf.divide(total_loss,tf.cast(N,tf.float32))

# define our optimizer
accuracy = tf.equal(tf.argmax(y_out, 1), tf.argmax(tf.one_hot(y,10), 1))
accuracy = tf.reduce_sum(tf.cast(accuracy,tf.float32))

optimizer = tf.train.AdamOptimizer(1e-2,0.85) # select optimizer and set learning rate
train_step = optimizer.minimize(mean_loss)


tf.summary.scalar("mean_loss", mean_loss)
tf.summary.scalar("accuracy", accuracy)
merged_summary_op = tf.summary.merge_all()

####################################################################################################

with tf.Session() as sess :
	with tf.device('/cpu:0'):
		tf.global_variables_initializer().run()	
		summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

		learning_rates = [1e-2,1e-3,1e-4,1e-5,1e-6]
		for i in learning_rates:
			for i in range(epoches):
				rand_index = np.random.choice(num_training, size=batch_size)
				loss,_,summary = sess.run([mean_loss,train_step,merged_summary_op],feed_dict={X:X_train[rand_index],y:Y_train[rand_index],train:1})
				summary_writer.add_summary(summary, i)
			val_acc = sess.run(accuracy,feed_dict={X:X_val,y:Y_val,train:0})
			print(val_acc)
