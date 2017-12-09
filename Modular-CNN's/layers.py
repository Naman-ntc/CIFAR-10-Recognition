import numpy as np
from utils import *

def affine-forward(x,w,b):
	"""
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    	
    cache is used to store data required in back-propogation
    """
    ###########################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You   #
    # will need to reshape the input into rows.                               #
    ###########################################################################
    w_shape = w.shape
    x_shape = x.shape
    D = w_shape[0]
    M = w_shape[1]
    N = x_shape[0]
    out = np.dot(x.reshape(N,D),w) + b
    cache = (x,w,b)
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return (out,cache)

def affine-backward(dout,cache):
	"""
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """    
    x = cache[0]
    w = cache[1]
    b = cache[2]
    N = x.shape[0]
    D = w.shape[0]
    dx = np.dot(dout,x.T).reshape(x.shape)
    dw = np.dot(x.reshape(N,D).T,dout)
    db = np.sum(dout,axis=0)

    return (dx,dw,db)

def activation_forward(x,activation):
	if activation=='ReLU':
		out = max(0,x)
	elif activation=='Leaky-ReLU':
		out = max(0.1*x,x)
	elif activation=='sigmoid':
		out = sigmoid(x)
	elif activation=='tanh':
		out = 0.5*(1+np.tanh(x))
	cache = x	
	return (out,cache)	
		
def activation_backward(dout,cache,activation):
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





