import numpy as np
from utils import *

def affine_forward(x,w,b):
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

def affine_backward(dout,cache):
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

def conv_forward(x,w,b,stride=[1,1],padding='VALID'):
	# x: Input data of shape (N, C, H, W)
	# w: Filter weights of shape (F, C, HH, WW)
	# b: Biases, of shape (F,)
	# stride: The number of pixels between adjacent receptive fields horizontally and vertically
	# padding: The number of pixels that will be used to zero-pad the input, defaults to valid

	# out: Output data, of shape (N, F, H', W') where H' and W' are given by
	# H' = 1 + (H + 2 * pad[0] - HH) / stride
    # W' = 1 + (W + 2 * pad[1] - WW) / stride
    # cache: (x, w, b, conv_param)

    if (padding=='VALID')
    	N,C,H,W = x.shape
    	F,C,HH,WW = w.shape
    	temp1 = 0
    	while((H-HH+2*temp1)%stride[0] != 0): 
    		temp1++
    	temp2 = 0
    	while((W-WW+2*temp2)%stride[1] != 0): 
    		temp2++
    	new_H = 1+(H-HH+2*temp1)/stride[0]
    	new_W = 1+(W-WW+2*temp2)/stride[1]
    	padding=[temp1,temp2]

    x_padded = np.zeros((N,C,H+2*stride[0],W+2*stride[1]))
   	x_padded[:,:,padding[0]:-1*padding[0],padding[1]:-1*padding[1]] = x

	out = np.zeros((N,F,new_H,new_W))

	for i in range(new_H):
		for j in range(new_W):
			for k in range(F):
				out[:,k,i,j] = np.sum(x_padded[:,:,i*stride[0],i*stride[0]+HH,j*stride[1],j*stride[1]+WW]*w[k,:,:,:],axis=(1,2,3))

	out = out + b[None,:,None,None]
	cache = (x,w,b,stride,padding)
	return (out,cache)

def conv_backward(dout,cache):
	x,w,b,stride,padding = cache
	N,C,H,W = x.shape
	F,C,HH,WW = w.shape
	new_H = 1+(H-HH+2*temp1)/stride[0]
	new_W = 1+(W-WW+2*temp2)/stride[1]
	x_padded = np.zeros((N,C,H+2*stride[0],W+2*stride[1]))
	x_padded[:,:,padding[0]:-1*padding[0],padding[1]:-1*padding[1]] = x
	dx_padded = np.zeros((N,C,H+2*padding[0],W+2*padding[1]))
	dw = np.zeros(w.shape)
	db = np.zeros(b.shape)

	for i in range(new_H):
		for j in range(new_W):
			
			for k in range(N):
				dx_padded[k,:,i*stride[0],i*stride[0]+HH,j*stride[1],j*stride[1]+WW] += np.sum((w[:, :, :, :]*(dout[n, :, i, j])[:,None ,None, None]),axis=0)
			for k in range(F):
				dw[k,:,:,:] += np.sum(x_padded[:,k,i*stride[0],i*stride[0]+HH,j*stride[1],j*stride[1]+WW]*dout[:,k,i,j],axis=0)

	dx = dx_padded[:,:,padding[0]:-1*padding[0],padding[1]:-1*padding[1]]
	return (dx,dw,db)

def maxpool_forward(x,pooling=[2,2],stride=[1,1],padding='VALID'):
	if (padding=='VALID')
    	N,C,H,W = x.shape
		HH,WW = pooling.shape
    	temp1 = 0
    	while((H-HH+2*temp1)%stride[0] != 0): 
    		temp1++
    	temp2 = 0
    	while((W-WW+2*temp2)%stride[1] != 0): 
    		temp2++
		new_H = 1+(H-HH+2*temp1)/stride[0]
		new_W = 1+(W-WW+2*temp2)/stride[1]
    	padding=[temp1,temp2]

    x_padded = np.zeros((N,C,H+2*stride[0],W+2*stride[1]))
    x_padded[:,:,padding[0]:-1*padding[0],padding[1]:-1*padding[1]] = x

	out = np.zeros((N,C,new_H,new_W))

	for i in range(new_H):
		for j in range(new_W):
			out[:,:,i,j] = np.max(x_padded[:,:,i*stride[0],i*stride[0]+HH,j*stride[1],j*stride[1]+WW],axis=(2,3))

	cache = (x,pooling,stride,padding)		
	return 	(out,cache)

def maxpool_backward(dout,cache):
	x,pooling,stride,padding = cache
	N,C,H,W = x.shape
	HH,WW = pooling.shape
	new_H = 1+(H-HH+2*temp1)/stride[0]
	new_W = 1+(W-WW+2*temp2)/stride[1]
	x_padded = np.zeros((N,C,H+2*stride[0],W+2*stride[1]))
	x_padded[:,:,padding[0]:-1*padding[0],padding[1]:-1*padding[1]] = x
	dx_padded = np.zeros((N,C,H+2*padding[0],W+2*padding[1]))

	for i in range(new_H):
		for j in range(new_W):
			dx_padded += dout[:,:,i,j]*((x_padded[:,:,i*stride[0]:i*stride[0]+HH,j*stride[1]:j*stride[1]+WW])==np.max((x_padded[:,:,i*stride[0]:i*stride[0]+HH,j*stride[1]:j*stride[1]+WW])[:,:,None,None],axis=(2,3)))

	dx = dx_padded[:,:,padding[0]:-1*padding[0],padding[1]:-1*padding[1]]
	return dx


