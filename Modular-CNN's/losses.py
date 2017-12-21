import numpy as np
from utils import *

def softmax_cross_entropy_loss(x,y_true):
	# x: [i,j] stores score for the socre for the jth class of the ith input
	# y_true: [i] stores class number for ith input numbered {0,1,2,......,k-1}
	N,K = x.shape
	y_true_onehot = np.zeros((N,K))
	y_true_onehot[np.arange(N),y_true] = 1

	x = x - np.max(x,axis=1,keepdims=True)
	
	log_probabilities = x - np.log(np.sum(np.exp(x),axis=1,keepdims=True))
	loss = -np.sum(y_true_onehot*log_probabilities)
	loss/=N

	probabilities = np.exp(log_probabilities)
	dx = probabilities
	dx[np.arange(N),y_true] -=1
	return (loss,dx)


def svm_loss(x,y_true):
	N,K = x.shape
	correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    dx = np.zeros((N,K))
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= np.sum(margins > 0, axis=1)
    dx /= N
    return (loss, dx)