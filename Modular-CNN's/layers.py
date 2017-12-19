import numpy as np
from utils import *

def affine-forward(x,w,b):
	# x is the input to the layer containing input data, of shape (N, d_1, ..., d_k)
    # w is array of weights of shape (D, M)
    # b is a array of biases of shape (M,)
    w_shape = w.shape
    x_shape = x.shape
    D = w_shape[0]
    M = w_shape[1]
    N = x_shape[0]
    # out: output, of shape (N, M)
    # cache: (x, w, b)
    out = np.dot(x.reshape(N,D),w) + b
    cache = (x,w,b)
    return (out,cache)

def affine-backward(dout,cache):
    # dout is upper-stream derivatives (derivatives of upper layer) of shape (N, M)
    # cache is tuple consisting of x and w 
    x = cache[0]
    w = cache[1]
    b = cache[2]
    N = x.shape[0]
    D = w.shape[0]
    dx = np.dot(dout,x.T).reshape(x.shape)
    dw = np.dot(x.reshape(N,D).T,dout)
    db = np.sum(dout,axis=0)

    return (dx,dw,db)

def activation_forward(x,activation='ReLU'):
	if activation=='ReLU':
		out = max(0,x)
	elif activation=='Leaky-ReLU':
		out = max(0.01*x,x)
	elif activation=='sigmoid':
		out = sigmoid(x)
	elif activation=='tanh':
		out = 0.5*(1+np.tanh(x))
	# elif activation=='PReLU':
	# 	out = max(args[0]*x,x)
	cache = x	
	return (out,cache)	
		
def activation_backward(dout,cache,activation='ReLU'):
	x = cache[0]
	if activation=='ReLU':
		dx = dout
		dx[x<=0] = 0
	elif activation=='Leaky-ReLU':
		dx[x>0] = 1
		dx[x<0] = 0.1
		dx[x==0] = 0
	elif activation=='sigmoid':
		dx = dout*sigmoid_prime(x)
	elif activation=='tanh':
		dx = 0.5*dout*tanh_prime(x)
	return dx	

def dropout_forward(x,keep_prob=0.5):
	mask = np.random.random_sample(x.shape) < keep_prob
	out = x*mask
	cache = (keep_prob,mask)
	return (out,cache)

def dropout_backward(dout,cache):
	mask = cache
	dx = dout*mask
	return dx

def conv_forward(x,w,b,strides=[1,1],padding='VALID'):
	# x: Input data of shape (N, C, H, W)
	# w: Filter weights of shape (F, C, HH, WW)
	# b: Biases, of shape (F,)
	# strides: The number of pixels between adjacent receptive fields horizontally and vertically
	# padding: The number of pixels that will be used to zero-pad the input, defaults to valid

	# H' = 1 + (H + 2 * pad[0] - HH) / stride
    # W' = 1 + (W + 2 * pad[1] - WW) / stride


    if (padding=='VALID')
    	_,_,H,W = x.shape
    	_,_,HH,WW = w.shape
    	temp1 = 0
    	while((H-HH+2*temp1)%stride[0] != 0): 
    		temp1++
    	temp2 = 0
    	while((W-WW+2*temp2)%stride[1] != 0): 
    		temp2++
    	padding=[temp1,temp2]		