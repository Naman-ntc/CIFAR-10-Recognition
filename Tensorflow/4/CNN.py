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

num_training=40000
num_validation=10000
num_test=10000


logs_path = '/home/naman/Repositories/CIFAR-10-Recognition/Tensorflow/examples/1'

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

##### convolution ==> batch_norm ===> activation ===> dropout
def test_output(X,y,is_training):
	avg_pooled = tf.layers.average_pooling2d(X,[3,3],[1,1])
	conved = tf.layers.conv2d(avg_pooled,64,[7,7])
	batch_normed = tf.layers.batch_normalization(conved,training=is_training)
	activated = tf.nn.leaky_relu(batch_normed,alpha=0.2)
	dropped = tf.layers.dropout(activated,training=is_training)
	conved = tf.layers.conv2d(dropped,32,[5,5],[2,2])
	batch_normed = tf.layers.batch_normalization(conved,training=is_training)
	activated = tf.nn.leaky_relu(batch_normed,alpha=0.15)
	max_pooled = tf.layers.max_pooling2d(activated,[2,2],[1,1])
	D = max_pooled.shape
	D = D[1]*D[2]*D[3]
	flattened = tf.reshape(max_pooled,[-1,D])
	densed = tf.layers.dense(flattened,2056)
	activated = tf.nn.selu(densed)
	dropped = tf.layers.dropout(activated,0.4,training=is_training)
	densed = tf.layers.dense(dropped,10)
	activated = tf.nn.relu(densed)
	y_out = activated
	#print(y_out.shape)
	return y_out

######################################################################################################

X = tf.placeholder(tf.float32,shape=[None,32,32,3])
y = tf.placeholder(tf.int32,shape=[None])
is_training = tf.placeholder(tf.bool)
lr = tf.placeholder(tf.float32)
N = tf.shape(X)[0]

y_out = test_output(X,y,is_training)

mean_loss = tf.losses.softmax_cross_entropy(logits=y_out, onehot_labels=tf.one_hot(y,10))
correct_prediction = tf.equal(tf.cast(tf.argmax(y_out,1),tf.int32), y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# define our optimizer
optimizer = tf.train.GradientDescentOptimizer(lr)

extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    updates = optimizer.minimize(mean_loss)

tf.summary.scalar("Mini Batch Loss", tf.multiply(mean_loss,tf.cast(N,tf.float32)))
tf.summary.scalar("accuracy", accuracy)
merged_summary_op = tf.summary.merge_all()

####################################################################################################

batch_size=64
epoches=6


extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
    train_step = optimizer.minimize(mean_loss)

with tf.Session() as sess :
	
	learning_rates = [1e-2,1e-4,1e-5,1e-6]
	for i in learning_rates:
		tf.global_variables_initializer().run()	
		for j in range(epoches):
			summary_writer = tf.summary.FileWriter(logs_path+str(j), graph=tf.get_default_graph())
			print("Epoch No. : %d"%(j))
			for k in range(801):
				rand_index = np.random.choice(num_training, size=batch_size)
				_,summary = sess.run([updates,merged_summary_op],feed_dict={X:X_train[rand_index],y:Y_train[rand_index],is_training:1,lr:i})
				summary_writer.add_summary(summary, k)
				if (k%100==0):
					curr_loss,curr_acc = sess.run([mean_loss,accuracy],feed_dict={X:X_train[rand_index],y:Y_train[rand_index],is_training:1,lr:i})
					print("Iteration %d : Mini Batch Loss = %.2f and accuracy = %.3f"%(k,batch_size*curr_loss,curr_acc))
		val_acc = sess.run(accuracy,feed_dict={X:X_val,y:Y_val,is_training:0,lr:0})
		print(val_acc)
